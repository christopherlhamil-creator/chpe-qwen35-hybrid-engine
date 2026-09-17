const std = @import("std");

pub const VEC_LEN_FP16: usize = 8; // 8x f16 = 128-bit NEON
pub const Vec8f16 = @Vector(VEC_LEN_FP16, f16);
pub const Vec4f32 = @Vector(4, f32);
pub const Vec16i8 = @Vector(16, i8);
pub const Vec4i32 = @Vector(4, i32);
pub const Vec16i32 = @Vector(16, i32);

/// Line-rate 8-lane NEON FP16 dot product with 4x unrolling (32 elements per cycle)
pub fn dotProductFp16Neon(a: []const f16, b: []const f16) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc0: Vec8f16 = @splat(0.0);
    var acc1: Vec8f16 = @splat(0.0);
    var acc2: Vec8f16 = @splat(0.0);
    var acc3: Vec8f16 = @splat(0.0);

    var i: usize = 0;
    while (i + 32 <= n) : (i += 32) {
        const va0: Vec8f16 = a[i .. i + 8][0..8].*;
        const vb0: Vec8f16 = b[i .. i + 8][0..8].*;
        acc0 = @mulAdd(Vec8f16, va0, vb0, acc0);

        const va1: Vec8f16 = a[i + 8 .. i + 16][0..8].*;
        const vb1: Vec8f16 = b[i + 8 .. i + 16][0..8].*;
        acc1 = @mulAdd(Vec8f16, va1, vb1, acc1);

        const va2: Vec8f16 = a[i + 16 .. i + 24][0..8].*;
        const vb2: Vec8f16 = b[i + 16 .. i + 24][0..8].*;
        acc2 = @mulAdd(Vec8f16, va2, vb2, acc2);

        const va3: Vec8f16 = a[i + 24 .. i + 32][0..8].*;
        const vb3: Vec8f16 = b[i + 24 .. i + 32][0..8].*;
        acc3 = @mulAdd(Vec8f16, va3, vb3, acc3);
    }

    // Accumulate the unrolled vectors
    const sum_vec = (acc0 + acc1) + (acc2 + acc3);
    var total: f32 = @reduce(.Add, @as(Vec4f32, @floatCast(sum_vec[0..4].*))) +
        @reduce(.Add, @as(Vec4f32, @floatCast(sum_vec[4..8].*)));

    // Handle tail elements
    while (i < n) : (i += 1) {
        total += @as(f32, a[i]) * @as(f32, b[i]);
    }

    return total;
}

/// NEON INT8 dot product with SDOT on ARMv8.2-A and portable SIMD fallback.
/// Eliminates intermediate horizontal reduction flushes, keeping 4 accumulators
/// in 128-bit vector registers to saturate dual ASIMD execution pipelines.
pub fn dotProductInt8Neon(a: []const i8, b: []const i8, scale_w: f32, scale_x: f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc: i32 = 0;
    var i: usize = 0;

    if (comptime @import("builtin").cpu.arch == .aarch64) {
        var v_acc0: Vec4i32 = @splat(0);
        var v_acc1: Vec4i32 = @splat(0);
        var v_acc2: Vec4i32 = @splat(0);
        var v_acc3: Vec4i32 = @splat(0);

        while (i + 64 <= n) : (i += 64) {
            const va0: Vec16i8 = a[i .. i + 16][0..16].*;
            const vb0: Vec16i8 = b[i .. i + 16][0..16].*;
            const va1: Vec16i8 = a[i + 16 .. i + 32][0..16].*;
            const vb1: Vec16i8 = b[i + 16 .. i + 32][0..16].*;
            const va2: Vec16i8 = a[i + 32 .. i + 48][0..16].*;
            const vb2: Vec16i8 = b[i + 32 .. i + 48][0..16].*;
            const va3: Vec16i8 = a[i + 48 .. i + 64][0..16].*;
            const vb3: Vec16i8 = b[i + 48 .. i + 64][0..16].*;

            asm (
                \\ sdot %[acc0].4s, %[va0].16b, %[vb0].16b
                \\ sdot %[acc1].4s, %[va1].16b, %[vb1].16b
                \\ sdot %[acc2].4s, %[va2].16b, %[vb2].16b
                \\ sdot %[acc3].4s, %[va3].16b, %[vb3].16b
                : [acc0] "+w" (v_acc0),
                  [acc1] "+w" (v_acc1),
                  [acc2] "+w" (v_acc2),
                  [acc3] "+w" (v_acc3),
                : [va0] "w" (va0),
                  [vb0] "w" (vb0),
                  [va1] "w" (va1),
                  [vb1] "w" (vb1),
                  [va2] "w" (va2),
                  [vb2] "w" (vb2),
                  [va3] "w" (va3),
                  [vb3] "w" (vb3),
            );
        }
        acc = @reduce(.Add, (v_acc0 + v_acc1) + (v_acc2 + v_acc3));
    } else {
        var v_acc0: Vec16i32 = @splat(0);
        var v_acc1: Vec16i32 = @splat(0);
        var v_acc2: Vec16i32 = @splat(0);
        var v_acc3: Vec16i32 = @splat(0);
        while (i + 64 <= n) : (i += 64) {
            const va0: Vec16i8 = a[i .. i + 16][0..16].*;
            const vb0: Vec16i8 = b[i .. i + 16][0..16].*;
            const va1: Vec16i8 = a[i + 16 .. i + 32][0..16].*;
            const vb1: Vec16i8 = b[i + 16 .. i + 32][0..16].*;
            const va2: Vec16i8 = a[i + 32 .. i + 48][0..16].*;
            const vb2: Vec16i8 = b[i + 32 .. i + 48][0..16].*;
            const va3: Vec16i8 = a[i + 48 .. i + 64][0..16].*;
            const vb3: Vec16i8 = b[i + 48 .. i + 64][0..16].*;

            v_acc0 += @as(Vec16i32, va0) * @as(Vec16i32, vb0);
            v_acc1 += @as(Vec16i32, va1) * @as(Vec16i32, vb1);
            v_acc2 += @as(Vec16i32, va2) * @as(Vec16i32, vb2);
            v_acc3 += @as(Vec16i32, va3) * @as(Vec16i32, vb3);
        }
        acc = @reduce(.Add, (v_acc0 + v_acc1) + (v_acc2 + v_acc3));
    }

    while (i < n) : (i += 1) {
        acc += @as(i32, a[i]) * @as(i32, b[i]);
    }

    return @as(f32, @floatFromInt(acc)) * (scale_w * scale_x);
}

/// Quad-Row INT8 Dot Product Kernel (4 rows evaluated against 1 activation vector).
/// Reuses activation vector `x` across all 4 weight rows, cutting L1d activation memory
/// traffic by 75% and saturating quad vector pipelines with 8 independent accumulator registers.
pub fn dotProductInt8QuadRowNeon(
    w0: []const i8,
    w1: []const i8,
    w2: []const i8,
    w3: []const i8,
    x: []const i8,
    scale_w: f32,
    scale_x: f32,
    out: *[4]f32,
) void {
    std.debug.assert(w0.len == x.len);
    std.debug.assert(w1.len == x.len);
    std.debug.assert(w2.len == x.len);
    std.debug.assert(w3.len == x.len);
    const n = x.len;
    var i: usize = 0;

    if (comptime @import("builtin").cpu.arch == .aarch64) {
        var a0_0: Vec4i32 = @splat(0);
        var a0_1: Vec4i32 = @splat(0);
        var a1_0: Vec4i32 = @splat(0);
        var a1_1: Vec4i32 = @splat(0);
        var a2_0: Vec4i32 = @splat(0);
        var a2_1: Vec4i32 = @splat(0);
        var a3_0: Vec4i32 = @splat(0);
        var a3_1: Vec4i32 = @splat(0);

        while (i + 32 <= n) : (i += 32) {
            const vx0: Vec16i8 = x[i .. i + 16][0..16].*;
            const vx1: Vec16i8 = x[i + 16 .. i + 32][0..16].*;

            const vw0_0: Vec16i8 = w0[i .. i + 16][0..16].*;
            const vw0_1: Vec16i8 = w0[i + 16 .. i + 32][0..16].*;
            const vw1_0: Vec16i8 = w1[i .. i + 16][0..16].*;
            const vw1_1: Vec16i8 = w1[i + 16 .. i + 32][0..16].*;
            const vw2_0: Vec16i8 = w2[i .. i + 16][0..16].*;
            const vw2_1: Vec16i8 = w2[i + 16 .. i + 32][0..16].*;
            const vw3_0: Vec16i8 = w3[i .. i + 16][0..16].*;
            const vw3_1: Vec16i8 = w3[i + 16 .. i + 32][0..16].*;

            asm (
                \\ sdot %[a0_0].4s, %[vw0_0].16b, %[vx0].16b
                \\ sdot %[a1_0].4s, %[vw1_0].16b, %[vx0].16b
                \\ sdot %[a2_0].4s, %[vw2_0].16b, %[vx0].16b
                \\ sdot %[a3_0].4s, %[vw3_0].16b, %[vx0].16b
                \\ sdot %[a0_1].4s, %[vw0_1].16b, %[vx1].16b
                \\ sdot %[a1_1].4s, %[vw1_1].16b, %[vx1].16b
                \\ sdot %[a2_1].4s, %[vw2_1].16b, %[vx1].16b
                \\ sdot %[a3_1].4s, %[vw3_1].16b, %[vx1].16b
                : [a0_0] "+w" (a0_0),
                  [a0_1] "+w" (a0_1),
                  [a1_0] "+w" (a1_0),
                  [a1_1] "+w" (a1_1),
                  [a2_0] "+w" (a2_0),
                  [a2_1] "+w" (a2_1),
                  [a3_0] "+w" (a3_0),
                  [a3_1] "+w" (a3_1),
                : [vw0_0] "w" (vw0_0),
                  [vw0_1] "w" (vw0_1),
                  [vw1_0] "w" (vw1_0),
                  [vw1_1] "w" (vw1_1),
                  [vw2_0] "w" (vw2_0),
                  [vw2_1] "w" (vw2_1),
                  [vw3_0] "w" (vw3_0),
                  [vw3_1] "w" (vw3_1),
                  [vx0] "w" (vx0),
                  [vx1] "w" (vx1),
            );
        }

        var acc0: i32 = @reduce(.Add, a0_0 + a0_1);
        var acc1: i32 = @reduce(.Add, a1_0 + a1_1);
        var acc2: i32 = @reduce(.Add, a2_0 + a2_1);
        var acc3: i32 = @reduce(.Add, a3_0 + a3_1);

        while (i < n) : (i += 1) {
            const xi = @as(i32, x[i]);
            acc0 += @as(i32, w0[i]) * xi;
            acc1 += @as(i32, w1[i]) * xi;
            acc2 += @as(i32, w2[i]) * xi;
            acc3 += @as(i32, w3[i]) * xi;
        }

        const norm = scale_w * scale_x;
        out[0] = @as(f32, @floatFromInt(acc0)) * norm;
        out[1] = @as(f32, @floatFromInt(acc1)) * norm;
        out[2] = @as(f32, @floatFromInt(acc2)) * norm;
        out[3] = @as(f32, @floatFromInt(acc3)) * norm;
    } else {
        var v_acc0: Vec16i32 = @splat(0);
        var v_acc1: Vec16i32 = @splat(0);
        var v_acc2: Vec16i32 = @splat(0);
        var v_acc3: Vec16i32 = @splat(0);

        while (i + 16 <= n) : (i += 16) {
            const vx: Vec16i32 = x[i .. i + 16][0..16].*;
            const vw0: Vec16i32 = w0[i .. i + 16][0..16].*;
            const vw1: Vec16i32 = w1[i .. i + 16][0..16].*;
            const vw2: Vec16i32 = w2[i .. i + 16][0..16].*;
            const vw3: Vec16i32 = w3[i .. i + 16][0..16].*;

            v_acc0 += vw0 * vx;
            v_acc1 += vw1 * vx;
            v_acc2 += vw2 * vx;
            v_acc3 += vw3 * vx;
        }

        var acc0: i32 = @reduce(.Add, v_acc0);
        var acc1: i32 = @reduce(.Add, v_acc1);
        var acc2: i32 = @reduce(.Add, v_acc2);
        var acc3: i32 = @reduce(.Add, v_acc3);

        while (i < n) : (i += 1) {
            const xi = @as(i32, x[i]);
            acc0 += @as(i32, w0[i]) * xi;
            acc1 += @as(i32, w1[i]) * xi;
            acc2 += @as(i32, w2[i]) * xi;
            acc3 += @as(i32, w3[i]) * xi;
        }

        const norm = scale_w * scale_x;
        out[0] = @as(f32, @floatFromInt(acc0)) * norm;
        out[1] = @as(f32, @floatFromInt(acc1)) * norm;
        out[2] = @as(f32, @floatFromInt(acc2)) * norm;
        out[3] = @as(f32, @floatFromInt(acc3)) * norm;
    }
}

/// Fused Gate + Up + SwiGLU Kernel.
/// Evaluates Gate row and Up row concurrently against activation vector `x`,
/// then directly evaluates SwiGLU activation in-register: out = silu(gate) * up.
/// Completely eliminates intermediate DRAM/cache writes and subsequent reads.
pub fn dotProductInt8GateUpSwiGLUNeon(
    w_gate: []const i8,
    w_up: []const i8,
    x: []const i8,
    scale_gate: f32,
    scale_up: f32,
    scale_x: f32,
) f32 {
    std.debug.assert(w_gate.len == x.len);
    std.debug.assert(w_up.len == x.len);
    const n = x.len;
    var i: usize = 0;

    var gate_val: f32 = 0.0;
    var up_val: f32 = 0.0;

    if (comptime @import("builtin").cpu.arch == .aarch64) {
        var ag_0: Vec4i32 = @splat(0);
        var ag_1: Vec4i32 = @splat(0);
        var au_0: Vec4i32 = @splat(0);
        var au_1: Vec4i32 = @splat(0);

        while (i + 32 <= n) : (i += 32) {
            const vx0: Vec16i8 = x[i .. i + 16][0..16].*;
            const vx1: Vec16i8 = x[i + 16 .. i + 32][0..16].*;

            const vg0: Vec16i8 = w_gate[i .. i + 16][0..16].*;
            const vg1: Vec16i8 = w_gate[i + 16 .. i + 32][0..16].*;

            const vu0: Vec16i8 = w_up[i .. i + 16][0..16].*;
            const vu1: Vec16i8 = w_up[i + 16 .. i + 32][0..16].*;

            asm (
                \\ sdot %[ag_0].4s, %[vg0].16b, %[vx0].16b
                \\ sdot %[au_0].4s, %[vu0].16b, %[vx0].16b
                \\ sdot %[ag_1].4s, %[vg1].16b, %[vx1].16b
                \\ sdot %[au_1].4s, %[vu1].16b, %[vx1].16b
                : [ag_0] "+w" (ag_0),
                  [ag_1] "+w" (ag_1),
                  [au_0] "+w" (au_0),
                  [au_1] "+w" (au_1),
                : [vg0] "w" (vg0),
                  [vg1] "w" (vg1),
                  [vu0] "w" (vu0),
                  [vu1] "w" (vu1),
                  [vx0] "w" (vx0),
                  [vx1] "w" (vx1),
            );
        }

        var acc_g: i32 = @reduce(.Add, ag_0 + ag_1);
        var acc_u: i32 = @reduce(.Add, au_0 + au_1);

        while (i < n) : (i += 1) {
            const xi = @as(i32, x[i]);
            acc_g += @as(i32, w_gate[i]) * xi;
            acc_u += @as(i32, w_up[i]) * xi;
        }

        gate_val = @as(f32, @floatFromInt(acc_g)) * (scale_gate * scale_x);
        up_val = @as(f32, @floatFromInt(acc_u)) * (scale_up * scale_x);
    } else {
        var vg_acc: Vec16i32 = @splat(0);
        var vu_acc: Vec16i32 = @splat(0);

        while (i + 16 <= n) : (i += 16) {
            const vx: Vec16i32 = x[i .. i + 16][0..16].*;
            const vg: Vec16i32 = w_gate[i .. i + 16][0..16].*;
            const vu: Vec16i32 = w_up[i .. i + 16][0..16].*;

            vg_acc += vg * vx;
            vu_acc += vu * vx;
        }

        var acc_g: i32 = @reduce(.Add, vg_acc);
        var acc_u: i32 = @reduce(.Add, vu_acc);

        while (i < n) : (i += 1) {
            const xi = @as(i32, x[i]);
            acc_g += @as(i32, w_gate[i]) * xi;
            acc_u += @as(i32, w_up[i]) * xi;
        }

        gate_val = @as(f32, @floatFromInt(acc_g)) * (scale_gate * scale_x);
        up_val = @as(f32, @floatFromInt(acc_u)) * (scale_up * scale_x);
    }

    const silu_g = gate_val / (1.0 + @exp(-gate_val));
    return silu_g * up_val;
}

/// Batch-4 Speculative Verification Dot Product Kernel.
/// Loads weight row `w` ONCE from RAM and multiplies against 4 candidate activation
/// vectors (`x0`, `x1`, `x2`, `x3`), yielding a 4x arithmetic intensity multiplier.
pub fn dotProductInt8Batch4Neon(
    w: []const i8,
    x0: []const i8,
    x1: []const i8,
    x2: []const i8,
    x3: []const i8,
    scale_w: f32,
    scales_x: [4]f32,
    out: *[4]f32,
) void {
    std.debug.assert(w.len == x0.len);
    const n = w.len;
    var i: usize = 0;

    if (comptime @import("builtin").cpu.arch == .aarch64) {
        var a0_0: Vec4i32 = @splat(0);
        var a0_1: Vec4i32 = @splat(0);
        var a1_0: Vec4i32 = @splat(0);
        var a1_1: Vec4i32 = @splat(0);
        var a2_0: Vec4i32 = @splat(0);
        var a2_1: Vec4i32 = @splat(0);
        var a3_0: Vec4i32 = @splat(0);
        var a3_1: Vec4i32 = @splat(0);

        while (i + 32 <= n) : (i += 32) {
            const vw0: Vec16i8 = w[i .. i + 16][0..16].*;
            const vw1: Vec16i8 = w[i + 16 .. i + 32][0..16].*;

            const vx0_0: Vec16i8 = x0[i .. i + 16][0..16].*;
            const vx0_1: Vec16i8 = x0[i + 16 .. i + 32][0..16].*;
            const vx1_0: Vec16i8 = x1[i .. i + 16][0..16].*;
            const vx1_1: Vec16i8 = x1[i + 16 .. i + 32][0..16].*;
            const vx2_0: Vec16i8 = x2[i .. i + 16][0..16].*;
            const vx2_1: Vec16i8 = x2[i + 16 .. i + 32][0..16].*;
            const vx3_0: Vec16i8 = x3[i .. i + 16][0..16].*;
            const vx3_1: Vec16i8 = x3[i + 16 .. i + 32][0..16].*;

            asm (
                \\ sdot %[a0_0].4s, %[vw0].16b, %[vx0_0].16b
                \\ sdot %[a0_1].4s, %[vw1].16b, %[vx0_1].16b
                \\ sdot %[a1_0].4s, %[vw0].16b, %[vx1_0].16b
                \\ sdot %[a1_1].4s, %[vw1].16b, %[vx1_1].16b
                \\ sdot %[a2_0].4s, %[vw0].16b, %[vx2_0].16b
                \\ sdot %[a2_1].4s, %[vw1].16b, %[vx2_1].16b
                \\ sdot %[a3_0].4s, %[vw0].16b, %[vx3_0].16b
                \\ sdot %[a3_1].4s, %[vw1].16b, %[vx3_1].16b
                : [a0_0] "+w" (a0_0),
                  [a0_1] "+w" (a0_1),
                  [a1_0] "+w" (a1_0),
                  [a1_1] "+w" (a1_1),
                  [a2_0] "+w" (a2_0),
                  [a2_1] "+w" (a2_1),
                  [a3_0] "+w" (a3_0),
                  [a3_1] "+w" (a3_1),
                : [vw0] "w" (vw0),
                  [vw1] "w" (vw1),
                  [vx0_0] "w" (vx0_0),
                  [vx0_1] "w" (vx0_1),
                  [vx1_0] "w" (vx1_0),
                  [vx1_1] "w" (vx1_1),
                  [vx2_0] "w" (vx2_0),
                  [vx2_1] "w" (vx2_1),
                  [vx3_0] "w" (vx3_0),
                  [vx3_1] "w" (vx3_1),
            );
        }

        var acc0: i32 = @reduce(.Add, a0_0 + a0_1);
        var acc1: i32 = @reduce(.Add, a1_0 + a1_1);
        var acc2: i32 = @reduce(.Add, a2_0 + a2_1);
        var acc3: i32 = @reduce(.Add, a3_0 + a3_1);

        while (i < n) : (i += 1) {
            const wi = @as(i32, w[i]);
            acc0 += wi * @as(i32, x0[i]);
            acc1 += wi * @as(i32, x1[i]);
            acc2 += wi * @as(i32, x2[i]);
            acc3 += wi * @as(i32, x3[i]);
        }

        out[0] = @as(f32, @floatFromInt(acc0)) * (scale_w * scales_x[0]);
        out[1] = @as(f32, @floatFromInt(acc1)) * (scale_w * scales_x[1]);
        out[2] = @as(f32, @floatFromInt(acc2)) * (scale_w * scales_x[2]);
        out[3] = @as(f32, @floatFromInt(acc3)) * (scale_w * scales_x[3]);
    } else {
        var v_acc0: Vec16i32 = @splat(0);
        var v_acc1: Vec16i32 = @splat(0);
        var v_acc2: Vec16i32 = @splat(0);
        var v_acc3: Vec16i32 = @splat(0);

        while (i + 16 <= n) : (i += 16) {
            const vw: Vec16i8 = w[i .. i + 16][0..16].*;
            const vx0: Vec16i8 = x0[i .. i + 16][0..16].*;
            const vx1: Vec16i8 = x1[i .. i + 16][0..16].*;
            const vx2: Vec16i8 = x2[i .. i + 16][0..16].*;
            const vx3: Vec16i8 = x3[i .. i + 16][0..16].*;

            const vw_32: Vec16i32 = vw;
            v_acc0 += vw_32 * @as(Vec16i32, vx0);
            v_acc1 += vw_32 * @as(Vec16i32, vx1);
            v_acc2 += vw_32 * @as(Vec16i32, vx2);
            v_acc3 += vw_32 * @as(Vec16i32, vx3);
        }

        var acc0: i32 = @reduce(.Add, v_acc0);
        var acc1: i32 = @reduce(.Add, v_acc1);
        var acc2: i32 = @reduce(.Add, v_acc2);
        var acc3: i32 = @reduce(.Add, v_acc3);

        while (i < n) : (i += 1) {
            const wi = @as(i32, w[i]);
            acc0 += wi * @as(i32, x0[i]);
            acc1 += wi * @as(i32, x1[i]);
            acc2 += wi * @as(i32, x2[i]);
            acc3 += wi * @as(i32, x3[i]);
        }

        out[0] = @as(f32, @floatFromInt(acc0)) * (scale_w * scales_x[0]);
        out[1] = @as(f32, @floatFromInt(acc1)) * (scale_w * scales_x[1]);
        out[2] = @as(f32, @floatFromInt(acc2)) * (scale_w * scales_x[2]);
        out[3] = @as(f32, @floatFromInt(acc3)) * (scale_w * scales_x[3]);
    }
}

/// Batch-8 Speculative Verification Dot Product Kernel.
/// Loads weight row `w` ONCE from RAM and multiplies against 8 candidate activation
/// vectors (`x0`..`x7`), yielding an 8x arithmetic intensity multiplier.
pub fn dotProductInt8Batch8Neon(
    w: []const i8,
    x0: []const i8,
    x1: []const i8,
    x2: []const i8,
    x3: []const i8,
    x4: []const i8,
    x5: []const i8,
    x6: []const i8,
    x7: []const i8,
    scale_w: f32,
    scales_x: [8]f32,
    out: *[8]f32,
) void {
    std.debug.assert(w.len == x0.len);
    const n = w.len;
    var i: usize = 0;

    if (comptime @import("builtin").cpu.arch == .aarch64) {
        var a0: Vec4i32 = @splat(0);
        var a1: Vec4i32 = @splat(0);
        var a2: Vec4i32 = @splat(0);
        var a3: Vec4i32 = @splat(0);
        var a4: Vec4i32 = @splat(0);
        var a5: Vec4i32 = @splat(0);
        var a6: Vec4i32 = @splat(0);
        var a7: Vec4i32 = @splat(0);

        while (i + 16 <= n) : (i += 16) {
            const vw0: Vec16i8 = w[i .. i + 16][0..16].*;
            const vx0: Vec16i8 = x0[i .. i + 16][0..16].*;
            const vx1: Vec16i8 = x1[i .. i + 16][0..16].*;
            const vx2: Vec16i8 = x2[i .. i + 16][0..16].*;
            const vx3: Vec16i8 = x3[i .. i + 16][0..16].*;
            const vx4: Vec16i8 = x4[i .. i + 16][0..16].*;
            const vx5: Vec16i8 = x5[i .. i + 16][0..16].*;
            const vx6: Vec16i8 = x6[i .. i + 16][0..16].*;
            const vx7: Vec16i8 = x7[i .. i + 16][0..16].*;

            asm (
                \\ sdot %[a0].4s, %[vw0].16b, %[vx0].16b
                \\ sdot %[a1].4s, %[vw0].16b, %[vx1].16b
                \\ sdot %[a2].4s, %[vw0].16b, %[vx2].16b
                \\ sdot %[a3].4s, %[vw0].16b, %[vx3].16b
                \\ sdot %[a4].4s, %[vw0].16b, %[vx4].16b
                \\ sdot %[a5].4s, %[vw0].16b, %[vx5].16b
                \\ sdot %[a6].4s, %[vw0].16b, %[vx6].16b
                \\ sdot %[a7].4s, %[vw0].16b, %[vx7].16b
                : [a0] "+w" (a0),
                  [a1] "+w" (a1),
                  [a2] "+w" (a2),
                  [a3] "+w" (a3),
                  [a4] "+w" (a4),
                  [a5] "+w" (a5),
                  [a6] "+w" (a6),
                  [a7] "+w" (a7),
                : [vw0] "w" (vw0),
                  [vx0] "w" (vx0),
                  [vx1] "w" (vx1),
                  [vx2] "w" (vx2),
                  [vx3] "w" (vx3),
                  [vx4] "w" (vx4),
                  [vx5] "w" (vx5),
                  [vx6] "w" (vx6),
                  [vx7] "w" (vx7),
            );
        }

        var acc0: i32 = @reduce(.Add, a0);
        var acc1: i32 = @reduce(.Add, a1);
        var acc2: i32 = @reduce(.Add, a2);
        var acc3: i32 = @reduce(.Add, a3);
        var acc4: i32 = @reduce(.Add, a4);
        var acc5: i32 = @reduce(.Add, a5);
        var acc6: i32 = @reduce(.Add, a6);
        var acc7: i32 = @reduce(.Add, a7);

        while (i < n) : (i += 1) {
            const wi = @as(i32, w[i]);
            acc0 += wi * @as(i32, x0[i]);
            acc1 += wi * @as(i32, x1[i]);
            acc2 += wi * @as(i32, x2[i]);
            acc3 += wi * @as(i32, x3[i]);
            acc4 += wi * @as(i32, x4[i]);
            acc5 += wi * @as(i32, x5[i]);
            acc6 += wi * @as(i32, x6[i]);
            acc7 += wi * @as(i32, x7[i]);
        }

        out[0] = @as(f32, @floatFromInt(acc0)) * (scale_w * scales_x[0]);
        out[1] = @as(f32, @floatFromInt(acc1)) * (scale_w * scales_x[1]);
        out[2] = @as(f32, @floatFromInt(acc2)) * (scale_w * scales_x[2]);
        out[3] = @as(f32, @floatFromInt(acc3)) * (scale_w * scales_x[3]);
        out[4] = @as(f32, @floatFromInt(acc4)) * (scale_w * scales_x[4]);
        out[5] = @as(f32, @floatFromInt(acc5)) * (scale_w * scales_x[5]);
        out[6] = @as(f32, @floatFromInt(acc6)) * (scale_w * scales_x[6]);
        out[7] = @as(f32, @floatFromInt(acc7)) * (scale_w * scales_x[7]);
    } else {
        var v_a0: Vec16i32 = @splat(0);
        var v_a1: Vec16i32 = @splat(0);
        var v_a2: Vec16i32 = @splat(0);
        var v_a3: Vec16i32 = @splat(0);
        var v_a4: Vec16i32 = @splat(0);
        var v_a5: Vec16i32 = @splat(0);
        var v_a6: Vec16i32 = @splat(0);
        var v_a7: Vec16i32 = @splat(0);

        while (i + 16 <= n) : (i += 16) {
            const vw: Vec16i8 = w[i .. i + 16][0..16].*;
            const vx0: Vec16i8 = x0[i .. i + 16][0..16].*;
            const vx1: Vec16i8 = x1[i .. i + 16][0..16].*;
            const vx2: Vec16i8 = x2[i .. i + 16][0..16].*;
            const vx3: Vec16i8 = x3[i .. i + 16][0..16].*;
            const vx4: Vec16i8 = x4[i .. i + 16][0..16].*;
            const vx5: Vec16i8 = x5[i .. i + 16][0..16].*;
            const vx6: Vec16i8 = x6[i .. i + 16][0..16].*;
            const vx7: Vec16i8 = x7[i .. i + 16][0..16].*;

            const vw_32: Vec16i32 = vw;
            v_a0 += vw_32 * @as(Vec16i32, vx0);
            v_a1 += vw_32 * @as(Vec16i32, vx1);
            v_a2 += vw_32 * @as(Vec16i32, vx2);
            v_a3 += vw_32 * @as(Vec16i32, vx3);
            v_a4 += vw_32 * @as(Vec16i32, vx4);
            v_a5 += vw_32 * @as(Vec16i32, vx5);
            v_a6 += vw_32 * @as(Vec16i32, vx6);
            v_a7 += vw_32 * @as(Vec16i32, vx7);
        }

        var acc0: i32 = @reduce(.Add, v_a0);
        var acc1: i32 = @reduce(.Add, v_a1);
        var acc2: i32 = @reduce(.Add, v_a2);
        var acc3: i32 = @reduce(.Add, v_a3);
        var acc4: i32 = @reduce(.Add, v_a4);
        var acc5: i32 = @reduce(.Add, v_a5);
        var acc6: i32 = @reduce(.Add, v_a6);
        var acc7: i32 = @reduce(.Add, v_a7);

        while (i < n) : (i += 1) {
            const wi = @as(i32, w[i]);
            acc0 += wi * @as(i32, x0[i]);
            acc1 += wi * @as(i32, x1[i]);
            acc2 += wi * @as(i32, x2[i]);
            acc3 += wi * @as(i32, x3[i]);
            acc4 += wi * @as(i32, x4[i]);
            acc5 += wi * @as(i32, x5[i]);
            acc6 += wi * @as(i32, x6[i]);
            acc7 += wi * @as(i32, x7[i]);
        }

        out[0] = @as(f32, @floatFromInt(acc0)) * (scale_w * scales_x[0]);
        out[1] = @as(f32, @floatFromInt(acc1)) * (scale_w * scales_x[1]);
        out[2] = @as(f32, @floatFromInt(acc2)) * (scale_w * scales_x[2]);
        out[3] = @as(f32, @floatFromInt(acc3)) * (scale_w * scales_x[3]);
        out[4] = @as(f32, @floatFromInt(acc4)) * (scale_w * scales_x[4]);
        out[5] = @as(f32, @floatFromInt(acc5)) * (scale_w * scales_x[5]);
        out[6] = @as(f32, @floatFromInt(acc6)) * (scale_w * scales_x[6]);
        out[7] = @as(f32, @floatFromInt(acc7)) * (scale_w * scales_x[7]);
    }
}

/// Dynamic activation symmetric quantization from float32 to int8
pub fn quantizeActivationInt8(x: []const f32, out_q8: []i8) f32 {
    std.debug.assert(x.len == out_q8.len);
    var max_val: f32 = 0.0;
    for (x) |v| {
        const av = @abs(v);
        if (av > max_val) max_val = av;
    }
    const scale = if (max_val > 1e-9) max_val / 127.0 else 1.0;
    const inv_scale = 1.0 / scale;

    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) {
        const xv: Vec4f32 = x[i .. i + 4][0..4].*;
        const scaled = xv * @as(Vec4f32, @splat(inv_scale));
        inline for (0..4) |j| {
            const rounded = @round(scaled[j]);
            const clamped = @max(-128.0, @min(127.0, rounded));
            out_q8[i + j] = @intFromFloat(clamped);
        }
    }
    while (i < x.len) : (i += 1) {
        const rounded = @round(x[i] * inv_scale);
        const clamped = @max(-128.0, @min(127.0, rounded));
        out_q8[i] = @intFromFloat(clamped);
    }
    return scale;
}

/// Root Mean Square Normalization: y = (x / sqrt(mean(x^2) + eps)) * gamma
pub fn rmsnorm(x: []const f32, gamma: []const f32, y: []f32, eps: f32) void {
    std.debug.assert(x.len == gamma.len);
    std.debug.assert(x.len == y.len);
    const n = x.len;

    var sum_sq_vec: Vec4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const vx: Vec4f32 = x[i .. i + 4][0..4].*;
        sum_sq_vec += vx * vx;
    }
    var sum_sq = @reduce(.Add, sum_sq_vec);
    while (i < n) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    const mean_sq = sum_sq / @as(f32, @floatFromInt(n));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    i = 0;
    const inv_rms_vec: Vec4f32 = @splat(inv_rms);
    while (i + 4 <= n) : (i += 4) {
        const vx: Vec4f32 = x[i .. i + 4][0..4].*;
        const vg: Vec4f32 = gamma[i .. i + 4][0..4].*;
        y[i .. i + 4][0..4].* = vx * inv_rms_vec * vg;
    }
    while (i < n) : (i += 1) {
        y[i] = x[i] * inv_rms * gamma[i];
    }
}

/// SwiGLU: out = silu(gate) * up = (gate / (1 + exp(-gate))) * up
pub fn siluMul(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(gate.len == out.len);
    for (gate, up, out) |g, u, *o| {
        const silu_g = g / (1.0 + @exp(-g));
        o.* = silu_g * u;
    }
}

test "neon dotProductInt8 symmetry" {
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8 };
    const b = [_]i8{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const res = dotProductInt8Neon(&a, &b, 1.0, 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), res);
}

test "activation int8 quantization" {
    const x = [_]f32{ -10.0, 0.0, 5.0, 10.0 };
    var q: [4]i8 = undefined;
    const scale = quantizeActivationInt8(&x, &q);
    try std.testing.expect(scale > 0.0);
    try std.testing.expectEqual(@as(i8, 127), q[3]);
    try std.testing.expectEqual(@as(i8, -127), q[0]);
}

test "neon dotProductInt8Batch4 equivalence" {
    var w: [64]i8 = undefined;
    var x0: [64]i8 = undefined;
    var x1: [64]i8 = undefined;
    var x2: [64]i8 = undefined;
    var x3: [64]i8 = undefined;

    for (0..64) |idx| {
        w[idx] = @intCast(@as(i32, @intCast(idx % 15)) - 7);
        x0[idx] = @intCast(@as(i32, @intCast(idx % 11)) - 5);
        x1[idx] = @intCast(@as(i32, @intCast((idx * 3) % 17)) - 8);
        x2[idx] = @intCast(@as(i32, @intCast((idx * 5) % 13)) - 6);
        x3[idx] = @intCast(@as(i32, @intCast((idx * 7) % 19)) - 9);
    }

    const scale_w: f32 = 0.05;
    const scales_x = [4]f32{ 0.01, 0.02, 0.03, 0.04 };

    var out_batch: [4]f32 = undefined;
    dotProductInt8Batch4Neon(&w, &x0, &x1, &x2, &x3, scale_w, scales_x, &out_batch);

    const ref0 = dotProductInt8Neon(&w, &x0, scale_w, scales_x[0]);
    const ref1 = dotProductInt8Neon(&w, &x1, scale_w, scales_x[1]);
    const ref2 = dotProductInt8Neon(&w, &x2, scale_w, scales_x[2]);
    const ref3 = dotProductInt8Neon(&w, &x3, scale_w, scales_x[3]);

    try std.testing.expectApproxEqAbs(ref0, out_batch[0], 1e-4);
    try std.testing.expectApproxEqAbs(ref1, out_batch[1], 1e-4);
    try std.testing.expectApproxEqAbs(ref2, out_batch[2], 1e-4);
    try std.testing.expectApproxEqAbs(ref3, out_batch[3], 1e-4);
}

test "neon dotProductInt8Batch8 equivalence" {
    var w: [64]i8 = undefined;
    var x: [8][64]i8 = undefined;
    for (0..64) |idx| {
        w[idx] = @intCast(@as(i32, @intCast(idx % 15)) - 7);
        for (0..8) |b| {
            const mult: usize = (b + 1) * 2 + 1;
            x[b][idx] = @intCast(@as(i32, @intCast((idx * mult) % 19)) - 9);
        }
    }

    const scale_w: f32 = 0.05;
    var scales_x: [8]f32 = undefined;
    for (0..8) |b| scales_x[b] = 0.01 * @as(f32, @floatFromInt(b + 1));

    var out_batch: [8]f32 = undefined;
    dotProductInt8Batch8Neon(&w, &x[0], &x[1], &x[2], &x[3], &x[4], &x[5], &x[6], &x[7], scale_w, scales_x, &out_batch);

    for (0..8) |b| {
        const ref = dotProductInt8Neon(&w, &x[b], scale_w, scales_x[b]);
        try std.testing.expectApproxEqAbs(ref, out_batch[b], 1e-4);
    }
}

test "neon dotProductInt8QuadRow equivalence" {
    var w0: [64]i8 = undefined;
    var w1: [64]i8 = undefined;
    var w2: [64]i8 = undefined;
    var w3: [64]i8 = undefined;
    var x: [64]i8 = undefined;

    for (0..64) |idx| {
        w0[idx] = @intCast(@as(i32, @intCast(idx % 15)) - 7);
        w1[idx] = @intCast(@as(i32, @intCast((idx * 3) % 17)) - 8);
        w2[idx] = @intCast(@as(i32, @intCast((idx * 5) % 13)) - 6);
        w3[idx] = @intCast(@as(i32, @intCast((idx * 7) % 19)) - 9);
        x[idx] = @intCast(@as(i32, @intCast((idx * 11) % 23)) - 11);
    }

    const scale_w: f32 = 0.05;
    const scale_x: f32 = 0.02;

    var out_quad: [4]f32 = undefined;
    dotProductInt8QuadRowNeon(&w0, &w1, &w2, &w3, &x, scale_w, scale_x, &out_quad);

    const ref0 = dotProductInt8Neon(&w0, &x, scale_w, scale_x);
    const ref1 = dotProductInt8Neon(&w1, &x, scale_w, scale_x);
    const ref2 = dotProductInt8Neon(&w2, &x, scale_w, scale_x);
    const ref3 = dotProductInt8Neon(&w3, &x, scale_w, scale_x);

    try std.testing.expectApproxEqAbs(ref0, out_quad[0], 1e-4);
    try std.testing.expectApproxEqAbs(ref1, out_quad[1], 1e-4);
    try std.testing.expectApproxEqAbs(ref2, out_quad[2], 1e-4);
    try std.testing.expectApproxEqAbs(ref3, out_quad[3], 1e-4);
}

test "neon dotProductInt8GateUpSwiGLU equivalence" {
    var wg: [64]i8 = undefined;
    var wu: [64]i8 = undefined;
    var x: [64]i8 = undefined;

    for (0..64) |idx| {
        wg[idx] = @intCast(@as(i32, @intCast(idx % 15)) - 7);
        wu[idx] = @intCast(@as(i32, @intCast((idx * 3) % 17)) - 8);
        x[idx] = @intCast(@as(i32, @intCast((idx * 5) % 13)) - 6);
    }

    const scale_g: f32 = 0.04;
    const scale_u: f32 = 0.06;
    const scale_x: f32 = 0.03;

    const fused_res = dotProductInt8GateUpSwiGLUNeon(&wg, &wu, &x, scale_g, scale_u, scale_x);

    const ref_g = dotProductInt8Neon(&wg, &x, scale_g, scale_x);
    const ref_u = dotProductInt8Neon(&wu, &x, scale_u, scale_x);
    const silu_g = ref_g / (1.0 + @exp(-ref_g));
    const ref_fused = silu_g * ref_u;

    try std.testing.expectApproxEqAbs(ref_fused, fused_res, 1e-4);
}
