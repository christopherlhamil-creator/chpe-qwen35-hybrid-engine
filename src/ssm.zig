const std = @import("std");

pub const HIDDEN_DIM: usize = 4096;
pub const CONV_KERNEL_DIM: usize = 4;
pub const LINEAR_KEY_HEAD_DIM: usize = 128;
pub const LINEAR_NUM_KEY_HEADS: usize = 16;
pub const LINEAR_VALUE_HEAD_DIM: usize = 128;
pub const LINEAR_NUM_VALUE_HEADS: usize = 32;

/// Circular ring buffer for Conv1D state (Kernel width K=4)
pub const Conv1DRingBuffer = struct {
    history: [CONV_KERNEL_DIM][HIDDEN_DIM]f32,
    head: usize,

    pub fn init() Conv1DRingBuffer {
        return .{
            .history = std.mem.zeroes([CONV_KERNEL_DIM][HIDDEN_DIM]f32),
            .head = 0,
        };
    }

    pub fn step(self: *Conv1DRingBuffer, input: *const [HIDDEN_DIM]f32) void {
        @memcpy(&self.history[self.head], input);
        self.head = (self.head + 1) % CONV_KERNEL_DIM;
    }

    pub fn convolve(
        self: *const Conv1DRingBuffer,
        kernel: *const [8192][CONV_KERNEL_DIM]f32,
        out: *[8192]f32,
    ) void {
        for (0..8192) |c| {
            var sum: f32 = 0.0;
            for (0..CONV_KERNEL_DIM) |k| {
                const idx = (self.head + CONV_KERNEL_DIM - 1 - k) % CONV_KERNEL_DIM;
                const in_val = self.history[idx][c % HIDDEN_DIM];
                sum += in_val * kernel[c][k];
            }
            out[c] = sum;
        }
    }
};

/// Recurrent State Matrix for 16 key heads x 128 x 128 values (1 MB per linear layer)
pub const LinearAttentionState = struct {
    // 16 heads * 128 * 128 * sizeof(f32) = 1,048,576 bytes
    state: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !LinearAttentionState {
        const total_elements = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM * LINEAR_VALUE_HEAD_DIM;
        const buf = try allocator.alloc(f32, total_elements);
        @memset(buf, 0.0);
        return .{
            .state = buf,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LinearAttentionState) void {
        self.allocator.free(self.state);
    }

    /// Single-step recurrent update: S_t = alpha * S_{t-1} + K_t^T * V_t
    pub fn update(
        self: *LinearAttentionState,
        decay: *const [LINEAR_NUM_KEY_HEADS]f32,
        k: *const [LINEAR_NUM_KEY_HEADS][LINEAR_KEY_HEAD_DIM]f32,
        v: *const [LINEAR_NUM_VALUE_HEADS][LINEAR_VALUE_HEAD_DIM]f32,
        q: *const [LINEAR_NUM_KEY_HEADS][LINEAR_KEY_HEAD_DIM]f32,
        out: *[HIDDEN_DIM]f32,
    ) void {
        const head_stride = LINEAR_KEY_HEAD_DIM * LINEAR_VALUE_HEAD_DIM;

        for (0..LINEAR_NUM_KEY_HEADS) |h| {
            const alpha = decay[h];
            const head_offset = h * head_stride;
            const v_head = v[h % LINEAR_NUM_VALUE_HEADS];

            for (0..LINEAR_KEY_HEAD_DIM) |i| {
                const ki = k[h][i];
                for (0..LINEAR_VALUE_HEAD_DIM) |j| {
                    const idx = head_offset + i * LINEAR_VALUE_HEAD_DIM + j;
                    self.state[idx] = self.state[idx] * alpha + ki * v_head[j];
                }
            }

            // Output projection from state: y = Q * S
            for (0..LINEAR_VALUE_HEAD_DIM) |j| {
                var acc: f32 = 0.0;
                for (0..LINEAR_KEY_HEAD_DIM) |i| {
                    const idx = head_offset + i * LINEAR_VALUE_HEAD_DIM + j;
                    acc += q[h][i] * self.state[idx];
                }
                const out_idx = h * (HIDDEN_DIM / LINEAR_NUM_KEY_HEADS) + (j % (HIDDEN_DIM / LINEAR_NUM_KEY_HEADS));
                out[out_idx] = acc;
            }
        }
    }
};
