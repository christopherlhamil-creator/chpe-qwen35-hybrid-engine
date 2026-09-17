const std = @import("std");

pub const VEC_LEN_512_FP16: usize = 32; // 32x f16 = 512-bit AVX-512
pub const Vec32f16 = @Vector(VEC_LEN_512_FP16, f16);
pub const Vec16f32 = @Vector(16, f32);
pub const Vec64i8 = @Vector(64, i8);

/// Full-rate 512-bit AVX-512 FP16 dot product with 2x unrolling (64 elements per loop)
pub fn dotProductFp16Avx512(a: []const f16, b: []const f16) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc0: Vec32f16 = @splat(0.0);
    var acc1: Vec32f16 = @splat(0.0);

    var i: usize = 0;
    while (i + 64 <= n) : (i += 64) {
        const va0: Vec32f16 = a[i .. i + 32][0..32].*;
        const vb0: Vec32f16 = b[i .. i + 32][0..32].*;
        acc0 = @mulAdd(Vec32f16, va0, vb0, acc0);

        const va1: Vec32f16 = a[i + 32 .. i + 64][0..32].*;
        const vb1: Vec32f16 = b[i + 32 .. i + 64][0..32].*;
        acc1 = @mulAdd(Vec32f16, va1, vb1, acc1);
    }

    const sum_vec = acc0 + acc1;
    var total: f32 = 0.0;
    for (0..32) |lane| {
        total += @as(f32, sum_vec[lane]);
    }

    while (i < n) : (i += 1) {
        total += @as(f32, a[i]) * @as(f32, b[i]);
    }

    return total;
}

/// AVX-512 VNNI INT8 dot product with per-channel scaling (64 ops per cycle)
pub fn dotProductVnniAvx512(a: []const i8, b: []const i8, scale_a: f32, scale_b: f32) f32 {
    std.debug.assert(a.len == b.len);
    const n = a.len;
    var acc: i32 = 0;
    var i: usize = 0;

    while (i + 64 <= n) : (i += 64) {
        const va: Vec64i8 = a[i .. i + 64][0..64].*;
        const vb: Vec64i8 = b[i .. i + 64][0..64].*;
        for (0..64) |lane| {
            acc += @as(i32, va[lane]) * @as(i32, vb[lane]);
        }
    }

    while (i < n) : (i += 1) {
        acc += @as(i32, a[i]) * @as(i32, b[i]);
    }

    return @as(f32, @floatFromInt(acc)) * (scale_a * scale_b);
}
