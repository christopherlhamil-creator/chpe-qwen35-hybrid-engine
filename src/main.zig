//! Executable CLI runner and benchmark suite for Qwen3.5-9B Hybrid .chpe Forward Decode.
//! Subsystem: chpe-qwen35-hybrid-engine/src/main.zig

const std = @import("std");
const engine_mod = @import("engine.zig");
const weight_archive = @import("weight_archive.zig");

fn printUsage() void {
    std.debug.print(
        \\CHPE Qwen3.5-9B Hybrid Inference Engine (ARMv8/v9 & x86_64)
        \\Usage: chpe_qwen9b [OPTIONS]
        \\
        \\Options:
        \\  --archive, --weights <path>  Path to Qwen3.5-9B-Base.q8.raw.chpe archive
        \\                               (default: models/Qwen3.5-9B-Base.q8.raw.chpe)
        \\  --bench <iters>              Number of benchmark iterations (default: 1)
        \\  --batch <1|4|8>              Batch size: 1 (Autoregressive), 4 or 8 (Speculative)
        \\  --token <id>                 Input token ID (default: 151644)
        \\  --telemetry <path>           Optional path to export JSON benchmark telemetry
        \\  --help, -h                   Show this help message
        \\
    , .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var archive_path: [:0]const u8 = "models/Qwen3.5-9B-Base.q8.raw.chpe";
    var bench_iters: usize = 1;
    var input_token: u32 = 151644; // Standard Qwen system start token
    var batch_size: usize = 1;
    var telemetry_path: ?[:0]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else if (std.mem.eql(u8, arg, "--archive") or std.mem.eql(u8, arg, "--weights")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                archive_path = s;
            }
        } else if (std.mem.eql(u8, arg, "--bench")) {
            if (i + 1 < args.len) {
                i += 1;
                bench_iters = try std.fmt.parseInt(usize, args[i], 10);
            }
        } else if (std.mem.eql(u8, arg, "--token")) {
            if (i + 1 < args.len) {
                i += 1;
                input_token = try std.fmt.parseInt(u32, args[i], 10);
            }
        } else if (std.mem.eql(u8, arg, "--batch")) {
            if (i + 1 < args.len) {
                i += 1;
                batch_size = try std.fmt.parseInt(usize, args[i], 10);
            }
        } else if (std.mem.eql(u8, arg, "--telemetry")) {
            if (i + 1 < args.len) {
                i += 1;
                const s = try allocator.allocSentinel(u8, args[i].len, 0);
                @memcpy(s, args[i]);
                telemetry_path = s;
            }
        }
    }

    std.debug.print("======================================================================\n", .{});
    std.debug.print("       CHPE QWEN3.5-9B ENTERPRISE CPU FORWARD ENGINE (8.95B HYBRID)   \n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("Archive Path : {s}\n", .{archive_path});
    std.debug.print("Bench Iters  : {d}\n", .{bench_iters});
    std.debug.print("Batch Size   : {d} {s}\n", .{ batch_size, if (batch_size == 8) "(Batch-8 Speculative Verification)" else if (batch_size == 4) "(Batch-4 Speculative Verification)" else "(Autoregressive)" });
    std.debug.print("Input Token  : {d}\n", .{input_token});
    std.debug.print("Architecture : 32 Layers (24 Linear Attention SSM + 8 Full GQA)\n", .{});
    std.debug.print("Hidden Dim   : {d} | MLP Dim: {d} | Vocab: {d}\n", .{ engine_mod.HIDDEN_DIM, engine_mod.INTERMEDIATE_DIM, engine_mod.VOCAB_SIZE });

    var archive = weight_archive.WeightArchive.openPosix(archive_path) catch |err| {
        std.debug.print("\n[ERROR] Failed to open model archive at '{s}': {s}\n", .{ archive_path, @errorName(err) });
        std.debug.print("Please fetch weights via: python3 scripts/fetch_weights.py\n", .{});
        return err;
    };
    defer archive.close();

    std.debug.print("Archive verified: {d} tiles ({d:.2} GB mmap'd)\n", .{ archive.tileCount(), @as(f64, @floatFromInt(archive.file_size)) / (1024.0 * 1024.0 * 1024.0) });

    var engine = try engine_mod.Qwen35Engine.init(allocator, &archive);
    defer engine.deinit();

    var latencies = try allocator.alloc(f64, bench_iters);
    defer allocator.free(latencies);

    var last_res: engine_mod.ForwardResult = undefined;

    if (batch_size == 8) {
        std.debug.print("\n[EXECUTION] Running {d} Batch-8 speculative verification step(s) with real weights...\n", .{bench_iters});

        for (0..bench_iters) |iter| {
            const tokens = [8]u32{
                input_token,
                @intCast((input_token +% 7) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 13) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 29) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 43) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 61) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 79) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 97) % engine_mod.VOCAB_SIZE),
            };

            const t_pass_start = engine_mod.nowNs();
            const batch_res = engine.stepBatch8(tokens);
            const pass_ns = engine_mod.nowNs() - t_pass_start;
            const cur_ms = @as(f64, @floatFromInt(pass_ns)) / 1_000_000.0;
            latencies[iter] = cur_ms;
            last_res = batch_res[0];

            const eff_tok_s = (8.0 * 1000.0) / cur_ms;
            std.debug.print("  -> Run {d}/{d} (Batch 8): {d:.2} ms total ({d:.2} ms / token, {d:.2} tok/s) | Argmax Tokens: [{d}, {d}, {d}, {d}, {d}, {d}, {d}, {d}]\n", .{
                iter + 1,
                bench_iters,
                cur_ms,
                cur_ms / 8.0,
                eff_tok_s,
                batch_res[0].argmax_token,
                batch_res[1].argmax_token,
                batch_res[2].argmax_token,
                batch_res[3].argmax_token,
                batch_res[4].argmax_token,
                batch_res[5].argmax_token,
                batch_res[6].argmax_token,
                batch_res[7].argmax_token,
            });

            input_token = batch_res[0].argmax_token;
        }
    } else if (batch_size == 4) {
        std.debug.print("\n[EXECUTION] Running {d} Batch-4 speculative verification step(s) with real weights...\n", .{bench_iters});

        for (0..bench_iters) |iter| {
            const tokens = [4]u32{
                input_token,
                @intCast((input_token +% 7) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 13) % engine_mod.VOCAB_SIZE),
                @intCast((input_token +% 29) % engine_mod.VOCAB_SIZE),
            };

            const t_pass_start = engine_mod.nowNs();
            const batch_res = engine.stepBatch4(tokens);
            const pass_ns = engine_mod.nowNs() - t_pass_start;
            const cur_ms = @as(f64, @floatFromInt(pass_ns)) / 1_000_000.0;
            latencies[iter] = cur_ms;
            last_res = batch_res[0];

            const eff_tok_s = (4.0 * 1000.0) / cur_ms;
            std.debug.print("  -> Run {d}/{d} (Batch 4): {d:.2} ms total ({d:.2} ms / token, {d:.2} tok/s) | Argmax Tokens: [{d}, {d}, {d}, {d}]\n", .{
                iter + 1,
                bench_iters,
                cur_ms,
                cur_ms / 4.0,
                eff_tok_s,
                batch_res[0].argmax_token,
                batch_res[1].argmax_token,
                batch_res[2].argmax_token,
                batch_res[3].argmax_token,
            });

            input_token = batch_res[0].argmax_token;
        }
    } else {
        std.debug.print("\n[EXECUTION] Running {d} forward decode step(s) with real weights...\n", .{bench_iters});

        for (0..bench_iters) |iter| {
            engine.embedToken(input_token);

            const res = engine.step();
            last_res = res;

            const cur_ms = @as(f64, @floatFromInt(res.elapsed_ns)) / 1_000_000.0;
            latencies[iter] = cur_ms;

            std.debug.print("  -> Run {d}/{d}: {d:.2} ms | Argmax Token: {d} (Logit: {d:.4}, Token0: {d:.4})\n", .{
                iter + 1,
                bench_iters,
                cur_ms,
                res.argmax_token,
                res.max_logit,
                res.token0_logit,
            });

            input_token = res.argmax_token;
        }
    }

    var min_ms: f64 = std.math.inf(f64);
    var max_ms: f64 = 0.0;
    var sum: f64 = 0.0;
    for (latencies) |lat| {
        if (lat < min_ms) min_ms = lat;
        if (lat > max_ms) max_ms = lat;
        sum += lat;
    }
    const mean_lat = sum / @as(f64, @floatFromInt(bench_iters));
    const tok_factor: f64 = if (batch_size == 8) 8000.0 else (if (batch_size == 4) 4000.0 else 1000.0);
    const throughput: f64 = if (mean_lat > 0) tok_factor / mean_lat else 0.0;
    const per_tok_ms: f64 = if (batch_size == 8) mean_lat / 8.0 else (if (batch_size == 4) mean_lat / 4.0 else mean_lat);

    std.debug.print("\n=== [MEASURED BENCHMARK SUMMARY] ===\n", .{});
    std.debug.print("Min Pass Latency   : {d:.2} ms\n", .{min_ms});
    std.debug.print("Mean Pass Latency  : {d:.2} ms\n", .{mean_lat});
    std.debug.print("Mean Token Latency : {d:.2} ms\n", .{per_tok_ms});
    std.debug.print("Max Pass Latency   : {d:.2} ms\n", .{max_ms});
    std.debug.print("Generation Speed   : {d:.3} tok/s\n", .{throughput});
    std.debug.print("Argmax Decoded ID  : {d}\n", .{last_res.argmax_token});
    std.debug.print("Maximum Vocab Logit: {d:.6}\n", .{last_res.max_logit});
    std.debug.print("Token 0 Logit      : {d:.6}\n", .{last_res.token0_logit});
    std.debug.print("Hidden Vector Norm : {d:.6}\n", .{last_res.hidden_norm});
    std.debug.print("Status             : {s}\n", .{if (last_res.all_finite) "SUCCESS (All logits finite)" else "FAIL (Non-finite logit detected)"});

    std.debug.print("\n--- [MICROARCHITECTURAL LATENCY BREAKDOWN (Last Run)] ---\n", .{});
    const qkv_ms = @as(f64, @floatFromInt(engine_mod.prof_qkv_ns)) / 1_000_000.0;
    const attn_ms = @as(f64, @floatFromInt(engine_mod.prof_attn_ns)) / 1_000_000.0;
    const norm_ms = @as(f64, @floatFromInt(engine_mod.prof_norm_ns)) / 1_000_000.0;
    const gateup_ms = @as(f64, @floatFromInt(engine_mod.prof_gateup_ns)) / 1_000_000.0;
    const down_ms = @as(f64, @floatFromInt(engine_mod.prof_down_ns)) / 1_000_000.0;
    const head_ms = @as(f64, @floatFromInt(engine_mod.prof_head_ns)) / 1_000_000.0;
    const total_prof_ms = qkv_ms + attn_ms + norm_ms + gateup_ms + down_ms + head_ms;
    std.debug.print("  SSM Proj (24 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ qkv_ms, if (total_prof_ms > 0) qkv_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  Attn GQA (8 layers)     : {d:6.2} ms ({d:4.1}%)\n", .{ attn_ms, if (total_prof_ms > 0) attn_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  RMSNorm  (65 norms)     : {d:6.2} ms ({d:4.1}%)\n", .{ norm_ms, if (total_prof_ms > 0) norm_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  Gate/Up  (32 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ gateup_ms, if (total_prof_ms > 0) gateup_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  Down     (32 layers)    : {d:6.2} ms ({d:4.1}%)\n", .{ down_ms, if (total_prof_ms > 0) down_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  LM Head  (248k vocab)   : {d:6.2} ms ({d:4.1}%)\n", .{ head_ms, if (total_prof_ms > 0) head_ms / total_prof_ms * 100.0 else 0 });
    std.debug.print("  Total Profiled Core Time: {d:6.2} ms\n", .{total_prof_ms});
    std.debug.print("======================================================================\n", .{});

    if (telemetry_path) |out_path| {
        const fd_val = std.os.linux.open(out_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
        if (std.os.linux.errno(fd_val) == .SUCCESS) {
            const fd: std.posix.fd_t = @intCast(fd_val);
            defer _ = std.os.linux.close(fd);

            var json_buf: [2048]u8 = undefined;
            const json_str = try std.fmt.bufPrint(&json_buf,
                \\{{
                \\  "model": "Qwen3.5-9B-Base",
                \\  "archive": "{s}",
                \\  "layers_executed": 32,
                \\  "benchmark_runs": {d},
                \\  "batch_size": {d},
                \\  "min_ms": {d:.2},
                \\  "mean_ms": {d:.2},
                \\  "max_ms": {d:.2},
                \\  "tokens_per_sec": {d:.3},
                \\  "argmax_token": {d},
                \\  "max_logit": {d:.6},
                \\  "token0_logit": {d:.6},
                \\  "hidden_norm": {d:.6},
                \\  "all_finite": {},
                \\  "status": "{s}"
                \\}}
                \\
            , .{
                archive_path,
                bench_iters,
                batch_size,
                min_ms,
                mean_lat,
                max_ms,
                throughput,
                last_res.argmax_token,
                last_res.max_logit,
                last_res.token0_logit,
                last_res.hidden_norm,
                last_res.all_finite,
                if (last_res.all_finite) "PASS" else "FAIL",
            });

            _ = std.os.linux.write(fd, json_str.ptr, json_str.len);
            std.debug.print("Telemetry written to {s}\n", .{out_path});
        }
    }
}
