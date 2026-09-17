//! Comptime Hardware Architecture & Kernel Execution Directives
//! Generated automatically by scripts/solve_ebm_hardware_gap.py
//! Verified by Z3 + Vampire + Leo-III + EBM solver stack.

pub const TARGET_HARDWARE = "ARMv9-A Google Axion (Neoverse-V2, 4 Cores @ 2.6 GHz / 3.0 GHz) - Qwen3.5-9B Q8";
pub const TARGET_ISA = "arm64";
pub const ACTIVE_CORES: usize = 4;
pub const VOCAB_ROWS_PER_CORE: usize = 62080;
pub const L1D_WORKING_BUDGET_BYTES: usize = 49152;
pub const TILE_WORKING_SET_BYTES: usize = 20480;

pub const FUSED_SWIGLU_WORKING_SET_BYTES: usize = 128;
pub const UNFUSED_TWOPASS_WORKING_SET_BYTES: usize = 98304;
pub const TILES_PER_GROUP: usize = 1;
pub const TOTAL_INTERMEDIATE_TILES: usize = 3072;
pub const TOTAL_INTERMEDIATE_GROUPS: usize = 3072;
pub const NUM_WORKERS: usize = 4;
pub const WORKER_TILE_STARTS: [4]usize = .{ 0, 768, 1536, 2304 };
pub const WORKER_TILE_ENDS: [4]usize = .{ 768, 1536, 2304, 3072 };
pub const WORKER_0_TILE_START: usize = 0;
pub const WORKER_0_TILE_END: usize = 768;
pub const WORKER_1_TILE_START: usize = 768;
pub const WORKER_1_TILE_END: usize = 1536;
pub const WORKER_2_TILE_START: usize = 1536;
pub const WORKER_2_TILE_END: usize = 2304;
pub const WORKER_3_TILE_START: usize = 2304;
pub const WORKER_3_TILE_END: usize = 3072;
pub const SDOT_VECTOR_REGS_USED: usize = 26;
pub const SDOT_VECTOR_REGS_BUDGET: usize = 32;

pub const SYMMETRICAL_ROW_ISOLATION: bool = true;
pub const SYMMETRICAL_BLOCK_SIZE: usize = 128;

pub const VECTOR_DOT_LANES: usize = 4;
pub const VECTOR_INSTRUCTION = "sdot.4s";
pub const VECTOR_WIDTH_BITS: usize = 128;
pub const UNROLL_ACCUMULATOR_FACTOR: usize = 4;
pub const REDUCTION_STRIDE: usize = 4096;
pub const USE_INT8_SDOT: bool = true;

pub const DRAM_BANDWIDTH_SATURATION_GB_S: f32 = 72.5000;
pub const TARGET_THROUGHPUT_TOK_S: f32 = 42.00;
pub const TARGET_DECODE_MS: f32 = 23.81;
pub const LATEST_MEASURED_DECODE_MS: f32 = 92.48;
pub const LATEST_MEASURED_TOK_S: f32 = 10.813;

// Upstream competitor baseline & remaining squeeze target
pub const LLAMA_BASELINE_MS: f32 = 210.00;
pub const LLAMA_BASELINE_TOK_S: f32 = 4.76;
pub const REMAINING_GAP_TO_LLAMA_MS: f32 = -117.52;
pub const SQUEEZED_TARGET_DECODE_MS: f32 = 64.38;
pub const SQUEEZED_TARGET_TOK_S: f32 = 15.53;

// Profiled layer breakdown (ms)
pub const PROFILE_GATE_UP_MS: f32 = 37.37;
pub const PROFILE_DOWN_PROJ_MS: f32 = 20.49;
pub const PROFILE_LM_HEAD_MS: f32 = 6.82;
pub const PROFILE_QKV_PROJ_MS: f32 = 5.14;
pub const PROFILE_O_PROJ_MS: f32 = 4.20;
pub const PROFILE_RMSNORM_MS: f32 = 0.46;
pub const PROFILE_ATTN_GQA_MS: f32 = 0.07;

// Squeeze targets (ms)
pub const TARGET_GATE_UP_MS: f32 = 34.00;
pub const TARGET_DOWN_PROJ_MS: f32 = 15.00;
pub const TARGET_LM_HEAD_MS: f32 = 5.50;

const std = @import("std");

test "hardware config parameters" {
    try std.testing.expectEqual(@as(usize, 4), ACTIVE_CORES);
    try std.testing.expectEqual(@as(usize, 49152), L1D_WORKING_BUDGET_BYTES);
    try std.testing.expect(SDOT_VECTOR_REGS_USED <= SDOT_VECTOR_REGS_BUDGET);
    try std.testing.expect(SQUEEZED_TARGET_DECODE_MS < LLAMA_BASELINE_MS);
}
