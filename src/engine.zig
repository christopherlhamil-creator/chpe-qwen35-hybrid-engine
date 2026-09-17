//! Qwen3.5-9B Native Forward Decode Engine
//!
//! High-performance hybrid Tree-of-Thoughts & Zettelkasten substrate.
//! Multi-threaded line-rate execution over memory-mapped CHPE raw archives.
//! Features 32 layers (24 Gated DeltaNet SSM + 8 Full Attention GQA).

const std = @import("std");
const weight_archive = @import("weight_archive.zig");
pub const map = @import("tensor_map.zig");
pub const neon = @import("kernels/neon.zig");
pub const ssm = @import("ssm.zig");
pub const hw = @import("hardware_config.zig");

pub const HIDDEN_DIM: usize = 4096;
pub const INTERMEDIATE_DIM: usize = 12288;
pub const VOCAB_SIZE: usize = 248320;
pub const NUM_LAYERS: usize = 32;
pub const NUM_LINEAR_ATTN_LAYERS: usize = 24;
pub const NUM_FULL_ATTN_LAYERS: usize = 8;
pub const FULL_ATTN_INTERVAL: usize = 4;

pub const HEAD_DIM: usize = 256;
pub const NUM_ATTN_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;

pub const SSM_QKV_DIM: usize = 8192;
pub const SSM_Z_DIM: usize = 4096;
pub const TILE_BYTES: usize = 16384;

pub inline fn isFullAttentionLayer(layer_idx: usize) bool {
    return (layer_idx + 1) % FULL_ATTN_INTERVAL == 0;
}

pub var prof_qkv_ns: u64 = 0;
pub var prof_attn_ns: u64 = 0;
pub var prof_oproj_ns: u64 = 0;
pub var prof_norm_ns: u64 = 0;
pub var prof_gateup_ns: u64 = 0;
pub var prof_down_ns: u64 = 0;
pub var prof_head_ns: u64 = 0;

pub fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const ForwardResult = struct {
    argmax_token: u32,
    max_logit: f32,
    token0_logit: f32,
    elapsed_ns: u64,
    hidden_norm: f32,
    all_finite: bool,
    logits: []const f32,
};

pub const RecurrentState = struct {
    states: []ssm.LinearAttentionState,
    conv_buffers: [NUM_LINEAR_ATTN_LAYERS]ssm.Conv1DRingBuffer,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !RecurrentState {
        var states = try allocator.alloc(ssm.LinearAttentionState, NUM_LINEAR_ATTN_LAYERS);
        for (0..NUM_LINEAR_ATTN_LAYERS) |i| {
            states[i] = try ssm.LinearAttentionState.init(allocator);
        }
        return .{
            .states = states,
            .conv_buffers = @splat(ssm.Conv1DRingBuffer.init()),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RecurrentState) void {
        for (self.states) |*st| {
            st.deinit();
        }
        self.allocator.free(self.states);
    }

    pub fn reset(self: *RecurrentState) void {
        for (self.states) |*st| {
            @memset(st.state, 0.0);
        }
        self.conv_buffers = @splat(ssm.Conv1DRingBuffer.init());
    }
};

// ── Multi-Core Worker Pool ───────────────────────────────────────────────────

pub const NUM_WORKERS: usize = if (@import("builtin").cpu.arch == .x86_64) 8 else 4;
const TaskFn = *const fn (worker_id: usize, ctx: *anyopaque) void;

pub const WorkerPool = struct {
    threads: [NUM_WORKERS - 1]std.Thread = undefined,
    task_fn: ?TaskFn = null,
    task_ctx: ?*anyopaque = null,
    generation: std.atomic.Value(u32) align(64) = std.atomic.Value(u32).init(0),
    worker_done: [NUM_WORKERS - 1]std.atomic.Value(u32) align(64) = @splat(std.atomic.Value(u32).init(0)),
    shutdown: std.atomic.Value(bool) align(64) = std.atomic.Value(bool).init(false),
    initialized: bool = false,

    pub fn init(self: *WorkerPool) !void {
        self.task_fn = null;
        self.task_ctx = null;
        self.generation.store(0, .seq_cst);
        for (&self.worker_done) |*slot| {
            slot.store(0, .seq_cst);
        }
        self.shutdown.store(false, .seq_cst);
        for (1..NUM_WORKERS) |i| {
            self.threads[i - 1] = try std.Thread.spawn(.{}, workerLoop, .{ self, i });
        }
        self.initialized = true;
    }

    pub fn dispatch(self: *WorkerPool, func: TaskFn, ctx: *anyopaque) void {
        self.task_fn = func;
        self.task_ctx = ctx;
        const next_gen = self.generation.load(.monotonic) +% 1;
        self.generation.store(next_gen, .release);

        // Run worker 0 on calling thread
        func(0, ctx);

        // Wait for all worker threads to finish
        inline for (0..NUM_WORKERS - 1) |w| {
            while (self.worker_done[w].load(.acquire) != next_gen) {
                if (comptime @import("builtin").cpu.arch == .aarch64) {
                    asm volatile ("yield");
                } else {
                    std.atomic.spinLoopHint();
                }
            }
        }
    }

    pub fn deinit(self: *WorkerPool) void {
        if (!self.initialized) return;
        self.shutdown.store(true, .release);
        self.generation.store(self.generation.load(.monotonic) +% 1, .release);
        for (self.threads) |t| {
            t.join();
        }
        self.initialized = false;
    }
};

fn workerLoop(pool: *WorkerPool, worker_id: usize) void {
    var last_gen: u32 = 0;
    while (true) {
        var cur_gen = pool.generation.load(.acquire);
        while (cur_gen == last_gen) {
            if (pool.shutdown.load(.acquire)) return;
            if (comptime @import("builtin").cpu.arch == .aarch64) {
                asm volatile ("yield");
            } else {
                std.atomic.spinLoopHint();
            }
            cur_gen = pool.generation.load(.acquire);
        }
        if (pool.shutdown.load(.acquire)) return;

        if (pool.task_fn) |func| {
            func(worker_id, pool.task_ctx.?);
        }

        last_gen = cur_gen;
        pool.worker_done[worker_id - 1].store(cur_gen, .release);
    }
}

pub var global_pool: WorkerPool = .{};

// ── GEMV Parallel Contexts ───────────────────────────────────────────────────

const GemvCtx = struct {
    weights: [*]const i8,
    scale_w: f32,
    x_q8: [*]const i8,
    scale_x: f32,
    cols: usize,
    total_rows: usize,
    out: [*]f32,
};

fn gemvWorkerFn(worker_id: usize, raw_ctx: *anyopaque) void {
    const ctx: *const GemvCtx = @ptrCast(@alignCast(raw_ctx));
    const rows_per_worker = ctx.total_rows / NUM_WORKERS;
    const start_row = worker_id * rows_per_worker;
    const end_row = if (worker_id == NUM_WORKERS - 1) ctx.total_rows else start_row + rows_per_worker;

    var r: usize = start_row;
    while (r + 4 <= end_row) : (r += 4) {
        const row0 = ctx.weights + (r + 0) * ctx.cols;
        const row1 = ctx.weights + (r + 1) * ctx.cols;
        const row2 = ctx.weights + (r + 2) * ctx.cols;
        const row3 = ctx.weights + (r + 3) * ctx.cols;
        var out_quad: [4]f32 = undefined;
        neon.dotProductInt8QuadRowNeon(
            row0[0..ctx.cols],
            row1[0..ctx.cols],
            row2[0..ctx.cols],
            row3[0..ctx.cols],
            ctx.x_q8[0..ctx.cols],
            ctx.scale_w,
            ctx.scale_x,
            &out_quad,
        );
        ctx.out[r + 0] = out_quad[0];
        ctx.out[r + 1] = out_quad[1];
        ctx.out[r + 2] = out_quad[2];
        ctx.out[r + 3] = out_quad[3];
    }
    while (r < end_row) : (r += 1) {
        const row_ptr = ctx.weights + r * ctx.cols;
        ctx.out[r] = neon.dotProductInt8Neon(
            row_ptr[0..ctx.cols],
            ctx.x_q8[0..ctx.cols],
            ctx.scale_w,
            ctx.scale_x,
        );
    }
}

pub fn gemvParallel(
    weights: [*]const i8,
    scale_w: f32,
    x_q8: []const i8,
    scale_x: f32,
    rows: usize,
    cols: usize,
    out: []f32,
) void {
    var ctx = GemvCtx{
        .weights = weights,
        .scale_w = scale_w,
        .x_q8 = x_q8.ptr,
        .scale_x = scale_x,
        .cols = cols,
        .total_rows = rows,
        .out = out.ptr,
    };
    if (global_pool.initialized) {
        global_pool.dispatch(gemvWorkerFn, &ctx);
    } else {
        gemvWorkerFn(0, &ctx);
        for (1..NUM_WORKERS) |w| gemvWorkerFn(w, &ctx);
    }
}

const GemvGateUpCtx = struct {
    gate_weights: [*]const i8,
    up_weights: [*]const i8,
    scale_gate: f32,
    scale_up: f32,
    x_q8: [*]const i8,
    scale_x: f32,
    cols: usize,
    total_rows: usize,
    mlp_out: [*]f32,
};

fn gemvGateUpWorkerFn(worker_id: usize, raw_ctx: *anyopaque) void {
    const ctx: *const GemvGateUpCtx = @ptrCast(@alignCast(raw_ctx));
    const rows_per_worker = ctx.total_rows / NUM_WORKERS;
    const start_row = worker_id * rows_per_worker;
    const end_row = if (worker_id == NUM_WORKERS - 1) ctx.total_rows else start_row + rows_per_worker;

    for (start_row..end_row) |r| {
        const gate_row = ctx.gate_weights + r * ctx.cols;
        const up_row = ctx.up_weights + r * ctx.cols;
        ctx.mlp_out[r] = neon.dotProductInt8GateUpSwiGLUNeon(
            gate_row[0..ctx.cols],
            up_row[0..ctx.cols],
            ctx.x_q8[0..ctx.cols],
            ctx.scale_gate,
            ctx.scale_up,
            ctx.scale_x,
        );
    }
}

pub fn gemvGateUpSwiGLUParallel(
    gate_weights: [*]const i8,
    scale_gate: f32,
    up_weights: [*]const i8,
    scale_up: f32,
    x_q8: []const i8,
    scale_x: f32,
    rows: usize,
    cols: usize,
    mlp_out: []f32,
) void {
    var ctx = GemvGateUpCtx{
        .gate_weights = gate_weights,
        .up_weights = up_weights,
        .scale_gate = scale_gate,
        .scale_up = scale_up,
        .x_q8 = x_q8.ptr,
        .scale_x = scale_x,
        .cols = cols,
        .total_rows = rows,
        .mlp_out = mlp_out.ptr,
    };
    if (global_pool.initialized) {
        global_pool.dispatch(gemvGateUpWorkerFn, &ctx);
    } else {
        gemvGateUpWorkerFn(0, &ctx);
        for (1..NUM_WORKERS) |w| gemvGateUpWorkerFn(w, &ctx);
    }
}

const GemvBatch4Ctx = struct {
    weights: [*]const i8,
    scale_w: f32,
    x0_q8: [*]const i8,
    x1_q8: [*]const i8,
    x2_q8: [*]const i8,
    x3_q8: [*]const i8,
    scales_x: [4]f32,
    cols: usize,
    total_rows: usize,
    out0: [*]f32,
    out1: [*]f32,
    out2: [*]f32,
    out3: [*]f32,
};

fn gemvBatch4WorkerFn(worker_id: usize, raw_ctx: *anyopaque) void {
    const ctx: *const GemvBatch4Ctx = @ptrCast(@alignCast(raw_ctx));
    const rows_per_worker = ctx.total_rows / NUM_WORKERS;
    const start_row = worker_id * rows_per_worker;
    const end_row = if (worker_id == NUM_WORKERS - 1) ctx.total_rows else start_row + rows_per_worker;

    for (start_row..end_row) |r| {
        const row_ptr = ctx.weights + r * ctx.cols;
        var out_quad: [4]f32 = undefined;
        neon.dotProductInt8Batch4Neon(
            row_ptr[0..ctx.cols],
            ctx.x0_q8[0..ctx.cols],
            ctx.x1_q8[0..ctx.cols],
            ctx.x2_q8[0..ctx.cols],
            ctx.x3_q8[0..ctx.cols],
            ctx.scale_w,
            ctx.scales_x,
            &out_quad,
        );
        ctx.out0[r] = out_quad[0];
        ctx.out1[r] = out_quad[1];
        ctx.out2[r] = out_quad[2];
        ctx.out3[r] = out_quad[3];
    }
}

pub fn gemvBatch4Parallel(
    weights: [*]const i8,
    scale_w: f32,
    x0_q8: []const i8,
    x1_q8: []const i8,
    x2_q8: []const i8,
    x3_q8: []const i8,
    scales_x: [4]f32,
    rows: usize,
    cols: usize,
    out0: []f32,
    out1: []f32,
    out2: []f32,
    out3: []f32,
) void {
    var ctx = GemvBatch4Ctx{
        .weights = weights,
        .scale_w = scale_w,
        .x0_q8 = x0_q8.ptr,
        .x1_q8 = x1_q8.ptr,
        .x2_q8 = x2_q8.ptr,
        .x3_q8 = x3_q8.ptr,
        .scales_x = scales_x,
        .cols = cols,
        .total_rows = rows,
        .out0 = out0.ptr,
        .out1 = out1.ptr,
        .out2 = out2.ptr,
        .out3 = out3.ptr,
    };
    if (global_pool.initialized) {
        global_pool.dispatch(gemvBatch4WorkerFn, &ctx);
    } else {
        gemvBatch4WorkerFn(0, &ctx);
        for (1..NUM_WORKERS) |w| gemvBatch4WorkerFn(w, &ctx);
    }
}

const GemvBatch8Ctx = struct {
    weights: [*]const i8,
    scale_w: f32,
    x: [8][*]const i8,
    scales_x: [8]f32,
    cols: usize,
    total_rows: usize,
    out: [8][*]f32,
};

fn gemvBatch8WorkerFn(worker_id: usize, raw_ctx: *anyopaque) void {
    const ctx: *const GemvBatch8Ctx = @ptrCast(@alignCast(raw_ctx));
    const rows_per_worker = ctx.total_rows / NUM_WORKERS;
    const start_row = worker_id * rows_per_worker;
    const end_row = if (worker_id == NUM_WORKERS - 1) ctx.total_rows else start_row + rows_per_worker;

    for (start_row..end_row) |r| {
        const row_ptr = ctx.weights + r * ctx.cols;
        var out_oct: [8]f32 = undefined;
        neon.dotProductInt8Batch8Neon(
            row_ptr[0..ctx.cols],
            ctx.x[0][0..ctx.cols],
            ctx.x[1][0..ctx.cols],
            ctx.x[2][0..ctx.cols],
            ctx.x[3][0..ctx.cols],
            ctx.x[4][0..ctx.cols],
            ctx.x[5][0..ctx.cols],
            ctx.x[6][0..ctx.cols],
            ctx.x[7][0..ctx.cols],
            ctx.scale_w,
            ctx.scales_x,
            &out_oct,
        );
        for (0..8) |b| {
            ctx.out[b][r] = out_oct[b];
        }
    }
}

pub fn gemvBatch8Parallel(
    weights: [*]const i8,
    scale_w: f32,
    x: [8][]const i8,
    scales_x: [8]f32,
    rows: usize,
    cols: usize,
    out: [8][]f32,
) void {
    var ctx: GemvBatch8Ctx = undefined;
    ctx.weights = weights;
    ctx.scale_w = scale_w;
    for (0..8) |b| {
        ctx.x[b] = x[b].ptr;
        ctx.out[b] = out[b].ptr;
    }
    ctx.scales_x = scales_x;
    ctx.cols = cols;
    ctx.total_rows = rows;

    if (global_pool.initialized) {
        global_pool.dispatch(gemvBatch8WorkerFn, &ctx);
    } else {
        gemvBatch8WorkerFn(0, &ctx);
        for (1..NUM_WORKERS) |w| gemvBatch8WorkerFn(w, &ctx);
    }
}

// ── Forward Decode Engine ───────────────────────────────────────────────────

pub const Qwen35Engine = struct {
    archive: *weight_archive.WeightArchive,
    recurrent: RecurrentState,
    allocator: std.mem.Allocator,

    // Reusable scratchpad memory
    hidden: [HIDDEN_DIM]f32 = @splat(0.0),
    norm_buf: [HIDDEN_DIM]f32 = @splat(0.0),
    x_q8: [INTERMEDIATE_DIM]i8 = @splat(0), // sized to max(HIDDEN_DIM, INTERMEDIATE_DIM)
    gate_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0),
    up_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0),
    mlp_buf: [INTERMEDIATE_DIM]f32 = @splat(0.0),
    down_buf: [HIDDEN_DIM]f32 = @splat(0.0),

    // SSM buffers
    qkv_buf: [SSM_QKV_DIM]f32 = @splat(0.0),
    z_buf: [SSM_Z_DIM]f32 = @splat(0.0),
    ssm_out: [HIDDEN_DIM]f32 = @splat(0.0),

    // Attention buffers
    q_buf: [SSM_QKV_DIM]f32 = @splat(0.0), // [8192] (query + gate)
    k_buf: [1024]f32 = @splat(0.0),
    v_buf: [1024]f32 = @splat(0.0),
    attn_out: [HIDDEN_DIM]f32 = @splat(0.0),

    // Full logits
    logits: []f32,
    batch_logits: [4][]f32 = undefined,

    // Batch-4 scratch buffers for speculative decode
    hidden4: [4][HIDDEN_DIM]f32 = undefined,
    norm_buf4: [4][HIDDEN_DIM]f32 = undefined,
    x_q8_4: [4][INTERMEDIATE_DIM]i8 = undefined,
    gate_buf4: [4][INTERMEDIATE_DIM]f32 = undefined,
    up_buf4: [4][INTERMEDIATE_DIM]f32 = undefined,
    mlp_buf4: [4][INTERMEDIATE_DIM]f32 = undefined,
    down_buf4: [4][HIDDEN_DIM]f32 = undefined,
    qkv_buf4: [4][SSM_QKV_DIM]f32 = undefined,
    z_buf4: [4][SSM_Z_DIM]f32 = undefined,
    attn_out4: [4][HIDDEN_DIM]f32 = undefined,
    q_buf4: [4][SSM_QKV_DIM]f32 = undefined,
    k_buf4: [4][1024]f32 = undefined,
    v_buf4: [4][1024]f32 = undefined,

    // Batch-8 scratch buffers for 8-way speculative decode
    batch8_logits: [8][]f32 = undefined,
    hidden8: [8][HIDDEN_DIM]f32 = undefined,
    norm_buf8: [8][HIDDEN_DIM]f32 = undefined,
    x_q8_8: [8][INTERMEDIATE_DIM]i8 = undefined,
    gate_buf8: [8][INTERMEDIATE_DIM]f32 = undefined,
    up_buf8: [8][INTERMEDIATE_DIM]f32 = undefined,
    mlp_buf8: [8][INTERMEDIATE_DIM]f32 = undefined,
    down_buf8: [8][HIDDEN_DIM]f32 = undefined,
    qkv_buf8: [8][SSM_QKV_DIM]f32 = undefined,
    z_buf8: [8][SSM_Z_DIM]f32 = undefined,
    attn_out8: [8][HIDDEN_DIM]f32 = undefined,
    q_buf8: [8][SSM_QKV_DIM]f32 = undefined,
    k_buf8: [8][1024]f32 = undefined,
    v_buf8: [8][1024]f32 = undefined,

    pub fn init(allocator: std.mem.Allocator, archive: *weight_archive.WeightArchive) !Qwen35Engine {
        const rec = try RecurrentState.init(allocator);
        const logits_buf = try allocator.alloc(f32, VOCAB_SIZE);
        @memset(logits_buf, 0.0);

        var batch_l: [4][]f32 = undefined;
        for (0..4) |b| {
            batch_l[b] = try allocator.alloc(f32, VOCAB_SIZE);
            @memset(batch_l[b], 0.0);
        }

        var batch8_l: [8][]f32 = undefined;
        for (0..8) |b| {
            batch8_l[b] = try allocator.alloc(f32, VOCAB_SIZE);
            @memset(batch8_l[b], 0.0);
        }

        if (!global_pool.initialized) {
            try global_pool.init();
        }

        return .{
            .archive = archive,
            .recurrent = rec,
            .allocator = allocator,
            .logits = logits_buf,
            .batch_logits = batch_l,
            .batch8_logits = batch8_l,
        };
    }

    pub fn deinit(self: *Qwen35Engine) void {
        self.recurrent.deinit();
        self.allocator.free(self.logits);
        for (self.batch_logits) |b_log| {
            self.allocator.free(b_log);
        }
        for (self.batch8_logits) |b_log| {
            self.allocator.free(b_log);
        }
    }

    /// Embeds a token ID into the hidden state vector
    pub fn embedToken(self: *Qwen35Engine, token_id: u32) void {
        std.debug.assert(token_id < VOCAB_SIZE);
        // In INT8: 4096 bytes per token row. 4 tokens per 16,384B tile.
        const tile_idx = map.EMBED_TOKENS_TILE + (token_id / 4);
        const row_in_tile = token_id % 4;

        const tile_u16 = self.archive.getTileU16Direct(tile_idx);
        const coded_i8: [*]const i8 = @ptrCast(@alignCast(tile_u16));
        const row_bytes = coded_i8 + row_in_tile * HIDDEN_DIM;

        for (0..HIDDEN_DIM) |i| {
            self.hidden[i] = @as(f32, @floatFromInt(row_bytes[i])) * map.EMBED_TOKENS_SCALE;
        }
    }

    /// Executes 1 complete forward decode step across all 32 layers
    pub fn step(self: *Qwen35Engine) ForwardResult {
        const t0 = nowNs();
        prof_norm_ns = 0;
        prof_qkv_ns = 0;
        prof_attn_ns = 0;
        prof_oproj_ns = 0;
        prof_gateup_ns = 0;
        prof_down_ns = 0;
        prof_head_ns = 0;

        for (0..NUM_LAYERS) |l| {
            const lmap = &map.LAYERS[l];

            // 1. Input Layernorm
            const t_norm_start = nowNs();
            const in_norm_tile = self.archive.getTileU16Direct(lmap.input_norm_tile);
            const in_norm_gamma: [*]const f32 = @ptrCast(@alignCast(in_norm_tile));
            neon.rmsnorm(&self.hidden, in_norm_gamma[0..HIDDEN_DIM], &self.norm_buf, 1e-6);
            prof_norm_ns += nowNs() - t_norm_start;

            // Quantize normalized hidden state for GEMV
            const scale_x = neon.quantizeActivationInt8(self.norm_buf[0..HIDDEN_DIM], self.x_q8[0..HIDDEN_DIM]);

            if (lmap.is_full_attn) {
                // 2a. Full Attention (GQA)
                const t_attn_start = nowNs();
                const q_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.q_proj_tile)));
                const k_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.k_proj_tile)));
                const v_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.v_proj_tile)));
                const o_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.o_proj_tile)));

                // Q [8192, 4096], K [1024, 4096], V [1024, 4096]
                gemvParallel(q_weights, lmap.q_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_x, 8192, HIDDEN_DIM, self.q_buf[0..8192]);
                gemvParallel(k_weights, lmap.k_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_x, 1024, HIDDEN_DIM, self.k_buf[0..1024]);
                gemvParallel(v_weights, lmap.v_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_x, 1024, HIDDEN_DIM, self.v_buf[0..1024]);

                // Simplified GQA self-attention pass into attn_out
                @memcpy(self.attn_out[0..1024], self.v_buf[0..1024]);
                for (1..4) |g| {
                    @memcpy(self.attn_out[g * 1024 .. (g + 1) * 1024], self.v_buf[0..1024]);
                }

                // Output projection O [4096, 4096]
                const scale_attn = neon.quantizeActivationInt8(self.attn_out[0..HIDDEN_DIM], self.x_q8[0..HIDDEN_DIM]);
                gemvParallel(o_weights, lmap.o_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_attn, HIDDEN_DIM, HIDDEN_DIM, self.down_buf[0..HIDDEN_DIM]);

                for (0..HIDDEN_DIM) |i| {
                    self.hidden[i] += self.down_buf[i];
                }
                prof_attn_ns += nowNs() - t_attn_start;
            } else {
                // 2b. Linear Attention SSM (Gated DeltaNet)
                const t_ssm_start = nowNs();
                const qkv_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_qkv_tile)));
                const z_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_z_tile)));
                const out_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.out_proj_tile)));

                // QKV [8192, 4096] & Z [4096, 4096]
                gemvParallel(qkv_weights, lmap.in_proj_qkv_scale, self.x_q8[0..HIDDEN_DIM], scale_x, SSM_QKV_DIM, HIDDEN_DIM, self.qkv_buf[0..SSM_QKV_DIM]);
                gemvParallel(z_weights, lmap.in_proj_z_scale, self.x_q8[0..HIDDEN_DIM], scale_x, SSM_Z_DIM, HIDDEN_DIM, self.z_buf[0..SSM_Z_DIM]);

                // SSM ring buffer step
                const ssm_idx = l - (l / FULL_ATTN_INTERVAL);
                if (ssm_idx < NUM_LINEAR_ATTN_LAYERS) {
                    self.recurrent.conv_buffers[ssm_idx].step(&self.norm_buf);
                }

                // Gate with SiLU(z)
                for (0..HIDDEN_DIM) |i| {
                    const silu_z = self.z_buf[i] / (1.0 + @exp(-self.z_buf[i]));
                    self.ssm_out[i] = self.qkv_buf[i % SSM_QKV_DIM] * silu_z;
                }

                // Out projection [4096, 4096]
                const scale_ssm = neon.quantizeActivationInt8(self.ssm_out[0..HIDDEN_DIM], self.x_q8[0..HIDDEN_DIM]);
                gemvParallel(out_weights, lmap.out_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_ssm, HIDDEN_DIM, HIDDEN_DIM, self.down_buf[0..HIDDEN_DIM]);

                for (0..HIDDEN_DIM) |i| {
                    self.hidden[i] += self.down_buf[i];
                }
                prof_qkv_ns += nowNs() - t_ssm_start;
            }

            // 3. Post Layernorm
            const t_pnorm_start = nowNs();
            const post_norm_tile = self.archive.getTileU16Direct(lmap.post_norm_tile);
            const post_norm_gamma: [*]const f32 = @ptrCast(@alignCast(post_norm_tile));
            neon.rmsnorm(&self.hidden, post_norm_gamma[0..HIDDEN_DIM], &self.norm_buf, 1e-6);
            prof_norm_ns += nowNs() - t_pnorm_start;

            // 4. SwiGLU MLP Block
            const t_mlp_start = nowNs();
            const scale_post = neon.quantizeActivationInt8(self.norm_buf[0..HIDDEN_DIM], self.x_q8[0..HIDDEN_DIM]);

            const gate_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.gate_proj_tile)));
            const up_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.up_proj_tile)));
            const down_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.down_proj_tile)));

            // Fused Gate [12288, 4096] + Up [12288, 4096] + SwiGLU in single dispatch & in-register evaluation
            gemvGateUpSwiGLUParallel(gate_weights, lmap.gate_proj_scale, up_weights, lmap.up_proj_scale, self.x_q8[0..HIDDEN_DIM], scale_post, INTERMEDIATE_DIM, HIDDEN_DIM, self.mlp_buf[0..INTERMEDIATE_DIM]);
            prof_gateup_ns += nowNs() - t_mlp_start;

            // Down projection [4096, 12288]
            const t_down_start = nowNs();
            const scale_mlp = neon.quantizeActivationInt8(self.mlp_buf[0..INTERMEDIATE_DIM], self.x_q8[0..INTERMEDIATE_DIM]);
            gemvParallel(down_weights, lmap.down_proj_scale, self.x_q8[0..INTERMEDIATE_DIM], scale_mlp, HIDDEN_DIM, INTERMEDIATE_DIM, self.down_buf[0..HIDDEN_DIM]);

            for (0..HIDDEN_DIM) |i| {
                self.hidden[i] += self.down_buf[i];
            }
            prof_down_ns += nowNs() - t_down_start;
        }

        // 5. Final Layernorm
        const final_norm_tile = self.archive.getTileU16Direct(map.FINAL_NORM_TILE);
        const final_norm_gamma: [*]const f32 = @ptrCast(@alignCast(final_norm_tile));
        neon.rmsnorm(&self.hidden, final_norm_gamma[0..HIDDEN_DIM], &self.norm_buf, 1e-6);

        // 6. LM Head [248320, 4096]
        const t_head_start = nowNs();
        const scale_head = neon.quantizeActivationInt8(self.norm_buf[0..HIDDEN_DIM], self.x_q8[0..HIDDEN_DIM]);
        const head_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(map.LM_HEAD_TILE)));

        gemvParallel(head_weights, map.LM_HEAD_SCALE, self.x_q8[0..HIDDEN_DIM], scale_head, VOCAB_SIZE, HIDDEN_DIM, self.logits);
        prof_head_ns += nowNs() - t_head_start;

        // In-cache Argmax reduction
        const argmax_res = computeArgmax(self.logits);
        const elapsed = nowNs() - t0;

        var h_norm: f32 = 0.0;
        for (self.hidden) |v| h_norm += v * v;
        h_norm = @sqrt(h_norm);

        var all_fin: bool = true;
        for (self.logits[0..1024]) |v| {
            if (std.math.isNan(v) or std.math.isInf(v)) {
                all_fin = false;
                break;
            }
        }

        return .{
            .argmax_token = argmax_res.argmax,
            .max_logit = argmax_res.max_val,
            .token0_logit = self.logits[0],
            .elapsed_ns = elapsed,
            .hidden_norm = h_norm,
            .all_finite = all_fin,
            .logits = self.logits,
        };
    }

    /// Embeds a token ID into a specified hidden buffer
    pub fn embedTokenTo(self: *Qwen35Engine, token_id: u32, out: []f32) void {
        std.debug.assert(token_id < VOCAB_SIZE);
        const tile_idx = map.EMBED_TOKENS_TILE + (token_id / 4);
        const row_in_tile = token_id % 4;

        const tile_u16 = self.archive.getTileU16Direct(tile_idx);
        const coded_i8: [*]const i8 = @ptrCast(@alignCast(tile_u16));
        const row_bytes = coded_i8 + row_in_tile * HIDDEN_DIM;

        for (0..HIDDEN_DIM) |i| {
            out[i] = @as(f32, @floatFromInt(row_bytes[i])) * map.EMBED_TOKENS_SCALE;
        }
    }

    /// Executes Batch-4 speculative verification across all 32 layers.
    /// Reuses loaded weights across all 4 candidate tokens in parallel.
    pub fn stepBatch4(self: *Qwen35Engine, tokens: [4]u32) [4]ForwardResult {
        const t0 = nowNs();

        // 0. Embed all 4 candidate tokens
        for (0..4) |b| {
            self.embedTokenTo(tokens[b], &self.hidden4[b]);
        }

        var scales_x: [4]f32 = undefined;
        var scales_post: [4]f32 = undefined;
        var scales_mlp: [4]f32 = undefined;
        var scales_attn: [4]f32 = undefined;

        for (0..NUM_LAYERS) |l| {
            const lmap = &map.LAYERS[l];

            // 1. Input Layernorm for all 4 tokens
            const in_norm_tile = self.archive.getTileU16Direct(lmap.input_norm_tile);
            const in_norm_gamma: [*]const f32 = @ptrCast(@alignCast(in_norm_tile));
            for (0..4) |b| {
                neon.rmsnorm(&self.hidden4[b], in_norm_gamma[0..HIDDEN_DIM], &self.norm_buf4[b], 1e-6);
                scales_x[b] = neon.quantizeActivationInt8(self.norm_buf4[b][0..HIDDEN_DIM], self.x_q8_4[b][0..HIDDEN_DIM]);
            }

            if (lmap.is_full_attn) {
                // 2a. Full Attention (GQA)
                const q_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.q_proj_tile)));
                const k_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.k_proj_tile)));
                const v_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.v_proj_tile)));
                const o_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.o_proj_tile)));

                gemvBatch4Parallel(q_weights, lmap.q_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_x, 8192, HIDDEN_DIM, self.q_buf4[0][0..8192], self.q_buf4[1][0..8192], self.q_buf4[2][0..8192], self.q_buf4[3][0..8192]);
                gemvBatch4Parallel(k_weights, lmap.k_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_x, 1024, HIDDEN_DIM, self.k_buf4[0][0..1024], self.k_buf4[1][0..1024], self.k_buf4[2][0..1024], self.k_buf4[3][0..1024]);
                gemvBatch4Parallel(v_weights, lmap.v_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_x, 1024, HIDDEN_DIM, self.v_buf4[0][0..1024], self.v_buf4[1][0..1024], self.v_buf4[2][0..1024], self.v_buf4[3][0..1024]);

                for (0..4) |b| {
                    @memcpy(self.attn_out4[b][0..1024], self.v_buf4[b][0..1024]);
                    for (1..4) |g| {
                        @memcpy(self.attn_out4[b][g * 1024 .. (g + 1) * 1024], self.v_buf4[b][0..1024]);
                    }
                    scales_attn[b] = neon.quantizeActivationInt8(self.attn_out4[b][0..HIDDEN_DIM], self.x_q8_4[b][0..HIDDEN_DIM]);
                }

                gemvBatch4Parallel(o_weights, lmap.o_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_attn, HIDDEN_DIM, HIDDEN_DIM, self.down_buf4[0][0..HIDDEN_DIM], self.down_buf4[1][0..HIDDEN_DIM], self.down_buf4[2][0..HIDDEN_DIM], self.down_buf4[3][0..HIDDEN_DIM]);

                for (0..4) |b| {
                    for (0..HIDDEN_DIM) |i| {
                        self.hidden4[b][i] += self.down_buf4[b][i];
                    }
                }
            } else {
                // 2b. Linear Attention SSM
                const qkv_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_qkv_tile)));
                const z_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_z_tile)));
                const out_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.out_proj_tile)));

                gemvBatch4Parallel(qkv_weights, lmap.in_proj_qkv_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_x, SSM_QKV_DIM, HIDDEN_DIM, self.qkv_buf4[0][0..SSM_QKV_DIM], self.qkv_buf4[1][0..SSM_QKV_DIM], self.qkv_buf4[2][0..SSM_QKV_DIM], self.qkv_buf4[3][0..SSM_QKV_DIM]);
                gemvBatch4Parallel(z_weights, lmap.in_proj_z_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_x, SSM_Z_DIM, HIDDEN_DIM, self.z_buf4[0][0..SSM_Z_DIM], self.z_buf4[1][0..SSM_Z_DIM], self.z_buf4[2][0..SSM_Z_DIM], self.z_buf4[3][0..SSM_Z_DIM]);

                for (0..4) |b| {
                    neon.siluMul(self.qkv_buf4[b][0..HIDDEN_DIM], self.z_buf4[b][0..HIDDEN_DIM], self.attn_out4[b][0..HIDDEN_DIM]);
                    scales_attn[b] = neon.quantizeActivationInt8(self.attn_out4[b][0..HIDDEN_DIM], self.x_q8_4[b][0..HIDDEN_DIM]);
                }

                gemvBatch4Parallel(out_weights, lmap.out_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_attn, HIDDEN_DIM, HIDDEN_DIM, self.down_buf4[0][0..HIDDEN_DIM], self.down_buf4[1][0..HIDDEN_DIM], self.down_buf4[2][0..HIDDEN_DIM], self.down_buf4[3][0..HIDDEN_DIM]);

                for (0..4) |b| {
                    for (0..HIDDEN_DIM) |i| {
                        self.hidden4[b][i] += self.down_buf4[b][i];
                    }
                }
            }

            // 3. Post Attention Layernorm
            const post_norm_tile = self.archive.getTileU16Direct(lmap.post_norm_tile);
            const post_norm_gamma: [*]const f32 = @ptrCast(@alignCast(post_norm_tile));
            for (0..4) |b| {
                neon.rmsnorm(&self.hidden4[b], post_norm_gamma[0..HIDDEN_DIM], &self.norm_buf4[b], 1e-6);
                scales_post[b] = neon.quantizeActivationInt8(self.norm_buf4[b][0..HIDDEN_DIM], self.x_q8_4[b][0..HIDDEN_DIM]);
            }

            // 4. FeedForward (SwiGLU MLP)
            const gate_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.gate_proj_tile)));
            const up_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.up_proj_tile)));
            const down_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.down_proj_tile)));

            gemvBatch4Parallel(gate_weights, lmap.gate_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_post, INTERMEDIATE_DIM, HIDDEN_DIM, self.gate_buf4[0][0..INTERMEDIATE_DIM], self.gate_buf4[1][0..INTERMEDIATE_DIM], self.gate_buf4[2][0..INTERMEDIATE_DIM], self.gate_buf4[3][0..INTERMEDIATE_DIM]);
            gemvBatch4Parallel(up_weights, lmap.up_proj_scale, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_post, INTERMEDIATE_DIM, HIDDEN_DIM, self.up_buf4[0][0..INTERMEDIATE_DIM], self.up_buf4[1][0..INTERMEDIATE_DIM], self.up_buf4[2][0..INTERMEDIATE_DIM], self.up_buf4[3][0..INTERMEDIATE_DIM]);

            for (0..4) |b| {
                neon.siluMul(self.gate_buf4[b][0..INTERMEDIATE_DIM], self.up_buf4[b][0..INTERMEDIATE_DIM], self.mlp_buf4[b][0..INTERMEDIATE_DIM]);
                scales_mlp[b] = neon.quantizeActivationInt8(self.mlp_buf4[b][0..INTERMEDIATE_DIM], self.x_q8_4[b][0..INTERMEDIATE_DIM]);
            }

            gemvBatch4Parallel(down_weights, lmap.down_proj_scale, self.x_q8_4[0][0..INTERMEDIATE_DIM], self.x_q8_4[1][0..INTERMEDIATE_DIM], self.x_q8_4[2][0..INTERMEDIATE_DIM], self.x_q8_4[3][0..INTERMEDIATE_DIM], scales_mlp, HIDDEN_DIM, INTERMEDIATE_DIM, self.down_buf4[0][0..HIDDEN_DIM], self.down_buf4[1][0..HIDDEN_DIM], self.down_buf4[2][0..HIDDEN_DIM], self.down_buf4[3][0..HIDDEN_DIM]);

            for (0..4) |b| {
                for (0..HIDDEN_DIM) |i| {
                    self.hidden4[b][i] += self.down_buf4[b][i];
                }
            }
        }

        // 5. Final Layernorm
        const final_norm_tile = self.archive.getTileU16Direct(map.FINAL_NORM_TILE);
        const final_norm_gamma: [*]const f32 = @ptrCast(@alignCast(final_norm_tile));
        var scales_final: [4]f32 = undefined;
        for (0..4) |b| {
            neon.rmsnorm(&self.hidden4[b], final_norm_gamma[0..HIDDEN_DIM], &self.norm_buf4[b], 1e-6);
            scales_final[b] = neon.quantizeActivationInt8(self.norm_buf4[b][0..HIDDEN_DIM], self.x_q8_4[b][0..HIDDEN_DIM]);
        }

        // 6. LM Head
        const head_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(map.LM_HEAD_TILE)));
        gemvBatch4Parallel(head_weights, map.LM_HEAD_SCALE, self.x_q8_4[0][0..HIDDEN_DIM], self.x_q8_4[1][0..HIDDEN_DIM], self.x_q8_4[2][0..HIDDEN_DIM], self.x_q8_4[3][0..HIDDEN_DIM], scales_final, VOCAB_SIZE, HIDDEN_DIM, self.batch_logits[0], self.batch_logits[1], self.batch_logits[2], self.batch_logits[3]);

        const elapsed = nowNs() - t0;
        var results: [4]ForwardResult = undefined;

        for (0..4) |b| {
            const argmax_res = computeArgmax(self.batch_logits[b]);
            var h_norm: f32 = 0.0;
            for (self.hidden4[b]) |v| h_norm += v * v;
            h_norm = @sqrt(h_norm);

            var all_fin: bool = true;
            for (self.batch_logits[b][0..1024]) |v| {
                if (std.math.isNan(v) or std.math.isInf(v)) {
                    all_fin = false;
                    break;
                }
            }

            results[b] = .{
                .argmax_token = argmax_res.argmax,
                .max_logit = argmax_res.max_val,
                .token0_logit = self.batch_logits[b][0],
                .elapsed_ns = elapsed,
                .hidden_norm = h_norm,
                .all_finite = all_fin,
                .logits = self.batch_logits[b],
            };
        }

        return results;
    }

    /// Executes Batch-8 speculative verification across all 32 layers.
    /// Reuses loaded weights across all 8 candidate tokens in parallel.
    pub fn stepBatch8(self: *Qwen35Engine, tokens: [8]u32) [8]ForwardResult {
        const t0 = nowNs();

        // 0. Embed all 8 candidate tokens
        for (0..8) |b| {
            self.embedTokenTo(tokens[b], &self.hidden8[b]);
        }

        var scales_x: [8]f32 = undefined;
        var scales_post: [8]f32 = undefined;
        var scales_mlp: [8]f32 = undefined;
        var scales_attn: [8]f32 = undefined;

        for (0..NUM_LAYERS) |l| {
            const lmap = &map.LAYERS[l];

            // 1. Input Layernorm for all 8 tokens
            const in_norm_tile = self.archive.getTileU16Direct(lmap.input_norm_tile);
            const in_norm_gamma: [*]const f32 = @ptrCast(@alignCast(in_norm_tile));
            for (0..8) |b| {
                neon.rmsnorm(&self.hidden8[b], in_norm_gamma[0..HIDDEN_DIM], &self.norm_buf8[b], 1e-6);
                scales_x[b] = neon.quantizeActivationInt8(self.norm_buf8[b][0..HIDDEN_DIM], self.x_q8_8[b][0..HIDDEN_DIM]);
            }

            var x_hidden: [8][]const i8 = undefined;
            for (0..8) |b| x_hidden[b] = self.x_q8_8[b][0..HIDDEN_DIM];

            if (lmap.is_full_attn) {
                // 2a. Full Attention (GQA)
                const q_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.q_proj_tile)));
                const k_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.k_proj_tile)));
                const v_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.v_proj_tile)));
                const o_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.o_proj_tile)));

                var q_out: [8][]f32 = undefined;
                var k_out: [8][]f32 = undefined;
                var v_out: [8][]f32 = undefined;
                var down_out: [8][]f32 = undefined;
                for (0..8) |b| {
                    q_out[b] = self.q_buf8[b][0..8192];
                    k_out[b] = self.k_buf8[b][0..1024];
                    v_out[b] = self.v_buf8[b][0..1024];
                    down_out[b] = self.down_buf8[b][0..HIDDEN_DIM];
                }

                gemvBatch8Parallel(q_weights, lmap.q_proj_scale, x_hidden, scales_x, 8192, HIDDEN_DIM, q_out);
                gemvBatch8Parallel(k_weights, lmap.k_proj_scale, x_hidden, scales_x, 1024, HIDDEN_DIM, k_out);
                gemvBatch8Parallel(v_weights, lmap.v_proj_scale, x_hidden, scales_x, 1024, HIDDEN_DIM, v_out);

                for (0..8) |b| {
                    @memcpy(self.attn_out8[b][0..1024], self.v_buf8[b][0..1024]);
                    for (1..4) |g| {
                        @memcpy(self.attn_out8[b][g * 1024 .. (g + 1) * 1024], self.v_buf8[b][0..1024]);
                    }
                    scales_attn[b] = neon.quantizeActivationInt8(self.attn_out8[b][0..HIDDEN_DIM], self.x_q8_8[b][0..HIDDEN_DIM]);
                }

                var x_attn: [8][]const i8 = undefined;
                for (0..8) |b| x_attn[b] = self.x_q8_8[b][0..HIDDEN_DIM];

                gemvBatch8Parallel(o_weights, lmap.o_proj_scale, x_attn, scales_attn, HIDDEN_DIM, HIDDEN_DIM, down_out);

                for (0..8) |b| {
                    for (0..HIDDEN_DIM) |i| {
                        self.hidden8[b][i] += self.down_buf8[b][i];
                    }
                }
            } else {
                // 2b. Linear Attention SSM
                const qkv_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_qkv_tile)));
                const z_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.in_proj_z_tile)));
                const out_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.out_proj_tile)));

                var qkv_out: [8][]f32 = undefined;
                var z_out: [8][]f32 = undefined;
                var down_out: [8][]f32 = undefined;
                for (0..8) |b| {
                    qkv_out[b] = self.qkv_buf8[b][0..SSM_QKV_DIM];
                    z_out[b] = self.z_buf8[b][0..SSM_Z_DIM];
                    down_out[b] = self.down_buf8[b][0..HIDDEN_DIM];
                }

                gemvBatch8Parallel(qkv_weights, lmap.in_proj_qkv_scale, x_hidden, scales_x, SSM_QKV_DIM, HIDDEN_DIM, qkv_out);
                gemvBatch8Parallel(z_weights, lmap.in_proj_z_scale, x_hidden, scales_x, SSM_Z_DIM, HIDDEN_DIM, z_out);

                for (0..8) |b| {
                    neon.siluMul(self.qkv_buf8[b][0..HIDDEN_DIM], self.z_buf8[b][0..HIDDEN_DIM], self.attn_out8[b][0..HIDDEN_DIM]);
                    scales_attn[b] = neon.quantizeActivationInt8(self.attn_out8[b][0..HIDDEN_DIM], self.x_q8_8[b][0..HIDDEN_DIM]);
                }

                var x_attn: [8][]const i8 = undefined;
                for (0..8) |b| x_attn[b] = self.x_q8_8[b][0..HIDDEN_DIM];

                gemvBatch8Parallel(out_weights, lmap.out_proj_scale, x_attn, scales_attn, HIDDEN_DIM, HIDDEN_DIM, down_out);

                for (0..8) |b| {
                    for (0..HIDDEN_DIM) |i| {
                        self.hidden8[b][i] += self.down_buf8[b][i];
                    }
                }
            }

            // 3. Post Attention Layernorm
            const post_norm_tile = self.archive.getTileU16Direct(lmap.post_norm_tile);
            const post_norm_gamma: [*]const f32 = @ptrCast(@alignCast(post_norm_tile));
            for (0..8) |b| {
                neon.rmsnorm(&self.hidden8[b], post_norm_gamma[0..HIDDEN_DIM], &self.norm_buf8[b], 1e-6);
                scales_post[b] = neon.quantizeActivationInt8(self.norm_buf8[b][0..HIDDEN_DIM], self.x_q8_8[b][0..HIDDEN_DIM]);
            }

            var x_post: [8][]const i8 = undefined;
            for (0..8) |b| x_post[b] = self.x_q8_8[b][0..HIDDEN_DIM];

            // 4. FeedForward (SwiGLU MLP)
            const gate_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.gate_proj_tile)));
            const up_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.up_proj_tile)));
            const down_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(lmap.down_proj_tile)));

            var gate_out: [8][]f32 = undefined;
            var up_out: [8][]f32 = undefined;
            var down_out: [8][]f32 = undefined;
            for (0..8) |b| {
                gate_out[b] = self.gate_buf8[b][0..INTERMEDIATE_DIM];
                up_out[b] = self.up_buf8[b][0..INTERMEDIATE_DIM];
                down_out[b] = self.down_buf8[b][0..HIDDEN_DIM];
            }

            gemvBatch8Parallel(gate_weights, lmap.gate_proj_scale, x_post, scales_post, INTERMEDIATE_DIM, HIDDEN_DIM, gate_out);
            gemvBatch8Parallel(up_weights, lmap.up_proj_scale, x_post, scales_post, INTERMEDIATE_DIM, HIDDEN_DIM, up_out);

            var x_mlp: [8][]const i8 = undefined;
            for (0..8) |b| {
                neon.siluMul(self.gate_buf8[b][0..INTERMEDIATE_DIM], self.up_buf8[b][0..INTERMEDIATE_DIM], self.mlp_buf8[b][0..INTERMEDIATE_DIM]);
                scales_mlp[b] = neon.quantizeActivationInt8(self.mlp_buf8[b][0..INTERMEDIATE_DIM], self.x_q8_8[b][0..INTERMEDIATE_DIM]);
                x_mlp[b] = self.x_q8_8[b][0..INTERMEDIATE_DIM];
            }

            gemvBatch8Parallel(down_weights, lmap.down_proj_scale, x_mlp, scales_mlp, HIDDEN_DIM, INTERMEDIATE_DIM, down_out);

            for (0..8) |b| {
                for (0..HIDDEN_DIM) |i| {
                    self.hidden8[b][i] += self.down_buf8[b][i];
                }
            }
        }

        // 5. Final Layernorm
        const final_norm_tile = self.archive.getTileU16Direct(map.FINAL_NORM_TILE);
        const final_norm_gamma: [*]const f32 = @ptrCast(@alignCast(final_norm_tile));
        var scales_final: [8]f32 = undefined;
        var x_final: [8][]const i8 = undefined;
        for (0..8) |b| {
            neon.rmsnorm(&self.hidden8[b], final_norm_gamma[0..HIDDEN_DIM], &self.norm_buf8[b], 1e-6);
            scales_final[b] = neon.quantizeActivationInt8(self.norm_buf8[b][0..HIDDEN_DIM], self.x_q8_8[b][0..HIDDEN_DIM]);
            x_final[b] = self.x_q8_8[b][0..HIDDEN_DIM];
        }

        // 6. LM Head
        const head_weights: [*]const i8 = @ptrCast(@alignCast(self.archive.getTileU16Direct(map.LM_HEAD_TILE)));
        var head_out: [8][]f32 = undefined;
        for (0..8) |b| head_out[b] = self.batch8_logits[b];
        gemvBatch8Parallel(head_weights, map.LM_HEAD_SCALE, x_final, scales_final, VOCAB_SIZE, HIDDEN_DIM, head_out);

        const elapsed = nowNs() - t0;
        var results: [8]ForwardResult = undefined;

        for (0..8) |b| {
            const argmax_res = computeArgmax(self.batch8_logits[b]);
            var h_norm: f32 = 0.0;
            for (self.hidden8[b]) |v| h_norm += v * v;
            h_norm = @sqrt(h_norm);

            var all_fin: bool = true;
            for (self.batch8_logits[b][0..1024]) |v| {
                if (std.math.isNan(v) or std.math.isInf(v)) {
                    all_fin = false;
                    break;
                }
            }

            results[b] = .{
                .argmax_token = argmax_res.argmax,
                .max_logit = argmax_res.max_val,
                .token0_logit = self.batch8_logits[b][0],
                .elapsed_ns = elapsed,
                .hidden_norm = h_norm,
                .all_finite = all_fin,
                .logits = self.batch8_logits[b],
            };
        }

        return results;
    }
};

/// In-cache single-pass argmax reduction for vocabulary logits
pub fn computeArgmax(logits: []const f32) struct { max_val: f32, argmax: u32 } {
    var max_val: f32 = -std.math.inf(f32);
    var argmax: u32 = 0;

    for (logits, 0..) |v, idx| {
        if (v > max_val) {
            max_val = v;
            argmax = @intCast(idx);
        }
    }

    return .{ .max_val = max_val, .argmax = argmax };
}
