r"""
Compile-time auto-tuning block: 

import torch
from math import inf, nan
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/vm/cvmzllqaiqz2jqqjj23ev3c5afbsxd2y3njfmlay7zqlaeotumum.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, marlin_gemm_1], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   marlin_gemm_1 => marlin_gemm_1
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg9_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg8_1 : Tensor "f16[3584][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float16), kwargs = {})
#   %mul_tensor_3 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg8_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "f16[s72, 37888][37888, 1]cuda:0"[num_users=2] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_3, None, %arg10_1, None, %arg11_1, None, None, %arg12_1, %arg13_1, %arg14_1, %arg15_1, 1125899906843648, %arg1_1, 37888, 3584, True, False, True, False), kwargs = {})
#   return %buf2,%buf3
triton_red_fused_fused_add_rms_norm_marlin_gemm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_marlin_gemm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_marlin_gemm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 234888192}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.176167936, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_marlin_gemm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 3584.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 3584*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/r4/cr4qgc5n35rkmdusnv5eqogl3zivf25ktbkbs4vr25qnhg45evun.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.silu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   marlin_gemm_2 => marlin_gemm_2
#   mul => mul_36
#   silu => add_30, convert_element_type, convert_element_type_1, div, exp, neg
# Graph fragment:
#   %marlin_gemm_1 : Tensor "f16[s72, 37888][37888, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "f16[s72, 18944][37888, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 18944), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type,), kwargs = {})
#   %exp : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_30 : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add_30), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %slice_2 : Tensor "f16[s72, 18944][37888, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 18944, 9223372036854775807), kwargs = {})
#   %mul_36 : Tensor "f16[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   %marlin_gemm_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_36, None, %arg16_1, None, %arg17_1, None, None, %arg18_1, %arg19_1, %arg20_1, %arg21_1, 1125899906843648, %arg1_1, 3584, 18944, True, False, True, False), kwargs = {})
#   return %buf6
triton_poi_fused_marlin_gemm_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_marlin_gemm_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_marlin_gemm_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1241513984}, 'kernel_num_gb': 0.931135488, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_marlin_gemm_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18944)
    x1 = xindex // 18944
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 37888*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (18944 + x0 + 37888*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/a3/ca3h5achxaszvoetv6qpq73yjdmjic2omyjsymw7xtaytwoaxpfv.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, marlin_gemm_3], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   marlin_gemm_3 => marlin_gemm_3
# Graph fragment:
#   %marlin_gemm_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %marlin_gemm : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg9_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %buf10 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf10]
#   %arg22_1 : Tensor "f16[3584][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.float16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg22_1), kwargs = {})
#   %marlin_gemm_3 : Tensor "f16[s72, 4608][4608, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_1, None, %arg23_1, %arg24_1, %arg25_1, None, None, %arg26_1, %arg27_1, %arg28_1, %arg29_1, 1125899906843648, %arg1_1, 4608, 3584, True, False, True, False), kwargs = {})
#   return %buf10,%convert_element_type_default_2,%buf11
triton_red_fused_fused_add_rms_norm_marlin_gemm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_marlin_gemm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_marlin_gemm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 411048960}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.293608448, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_marlin_gemm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 3584*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 3584.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 3584*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/jy/cjys7nryl5rskdjzlfm4ledjujmflsgdzvh2nlm6dp6ggghuod52.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 128)
        x1 = ((xindex // 128) % 4)
        x2 = xindex // 512
        x4 = xindex
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 64, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 32768")
        tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp7 * tmp14
        tmp16 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 - tmp18
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp6, tmp19, tmp20)
        tmp22 = tmp0 >= tmp5
        tmp23 = tl.full([1], 128, tl.int64)
        tmp24 = tmp0 < tmp23
        tmp25 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp26 < 0
        tmp30 = tl.where(tmp29, tmp28, tmp26)
        tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
        tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp25 * tmp32
        tmp34 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp33 + tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp22, tmp37, tmp38)
        tmp40 = tl.where(tmp6, tmp21, tmp39)
        tl.store(out_ptr0 + (x4), tmp40, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 128)
        x6 = ((xindex // 128) % 28)
        x7 = xindex // 3584
        x9 = xindex
        tmp41 = (x5).to(tl.int32)
        tmp42 = tl.full([1], 0, tl.int64)
        tmp43 = tmp41 >= tmp42
        tmp44 = (x5).to(tl.int64)
        tmp45 = (tmp44).to(tl.int64)
        tmp46 = tl.full([1], 64, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr1 + (x7), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp49 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp49)
        tl.device_assert(((0 <= tl.broadcast_to(tmp53, [XBLOCK])) & (tl.broadcast_to(tmp53, [XBLOCK]) < 32768)) | ~(tmp47 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp53, [XBLOCK]) < 32768")
        tmp55 = tl.load(in_ptr2 + (128*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp48 * tmp55
        tmp57 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr2 + (64 + 128*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 - tmp59
        tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
        tmp62 = tl.where(tmp47, tmp60, tmp61)
        tmp63 = tmp41 >= tmp46
        tmp64 = tl.full([1], 128, tl.int64)
        tmp65 = tmp41 < tmp64
        tmp66 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x7), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp69 = tmp67 + tmp68
        tmp70 = tmp67 < 0
        tmp71 = tl.where(tmp70, tmp69, tmp67)
        tl.device_assert(((0 <= tl.broadcast_to(tmp71, [XBLOCK])) & (tl.broadcast_to(tmp71, [XBLOCK]) < 32768)) | ~(tmp63 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp71, [XBLOCK]) < 32768")
        tmp73 = tl.load(in_ptr2 + (128*tmp71 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp66 * tmp73
        tmp75 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + (64 + 128*tmp71 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tmp75 * tmp76
        tmp78 = tmp74 + tmp77
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp63, tmp78, tmp79)
        tmp81 = tl.where(tmp47, tmp62, tmp80)
        tl.store(out_ptr1 + (x9), tmp81, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((8192, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((8192, 28, 128), (3584, 128, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 29360128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.352452608
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    raw_stream0 = get_raw_stream(0)
raw_stream0 = get_raw_stream(0)
buf1 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
arg9_1 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
arg8_1 = generate_example_value((3584,), (1,), 'cuda:0', torch.float16, 0, (3584,))
buf3 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_marlin_gemm_0.run(buf1, arg9_1, arg8_1, buf3, 8192, 3584, stream=raw_stream0)
del arg8_1, buf3

raw_stream0 = get_raw_stream(0)
buf5 = generate_example_value((8192, 37888), (37888, 1), 'cuda:0', torch.float16, 0, (8192, 37888))
buf6 = generate_example_value((8192, 18944), (18944, 1), 'cuda:0', torch.float16, 0, (8192, 18944))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_marlin_gemm_mul_silu_slice_1.run(buf5, buf6, 155189248, stream=raw_stream0)
del buf5, buf6

raw_stream0 = get_raw_stream(0)
buf8 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
arg22_1 = generate_example_value((3584,), (1,), 'cuda:0', torch.float16, 0, (3584,))
buf9 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
buf11 = generate_example_value((8192, 3584), (3584, 1), 'cuda:0', torch.float16, 0, (8192, 3584))
with torch.cuda._DeviceGuard(0):
    triton_red_fused_fused_add_rms_norm_marlin_gemm_2.run(buf8, buf1, arg9_1, arg22_1, buf9, buf11, 8192, 3584, stream=raw_stream0)
del buf1, arg9_1, buf8, arg22_1, buf9, buf11

raw_stream0 = get_raw_stream(0)
buf13 = generate_example_value((8192, 4608), (4608, 1), 'cuda:0', torch.float16, 0, (8192, 4608))
arg30_1 = generate_example_value((8192,), (1,), 'cuda:0', torch.int64, 0, (8192,))
arg31_1 = generate_example_value((32768, 128), (128, 1), 'cuda:0', torch.float16, 0, (32768, 128))
buf14 = generate_example_value((8192, 4, 128), (512, 128, 1), 'cuda:0', torch.float16, 0, (8192, 4, 128))
buf15 = generate_example_value((8192, 28, 128), (3584, 128, 1), 'cuda:0', torch.float16, 0, (8192, 28, 128))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_3.run(buf13, arg30_1, arg31_1, buf14, buf15, 4194304, 29360128, stream=raw_stream0)
del buf13, arg30_1, arg31_1, buf14, buf15

"""
# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/vm/cvmzllqaiqz2jqqjj23ev3c5afbsxd2y3njfmlay7zqlaeotumum.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, marlin_gemm_1], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, add_tensor_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mean_dim_1, mul_tensor_2, mul_tensor_3, pow_tensor_scalar_1, rsqrt_default_1
#   marlin_gemm_1 => marlin_gemm_1
# Graph fragment:
#   %marlin_gemm : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg9_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %buf2 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf2]
#   %arg8_1 : Tensor "f16[3584][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %pow_tensor_scalar_1 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor_2, 2), kwargs = {})
#   %mean_dim_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar_1, [-1], True), kwargs = {})
#   %add_tensor_3 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim_1, 1e-06), kwargs = {})
#   %rsqrt_default_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_3,), kwargs = {})
#   %mul_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor_2, %rsqrt_default_1), kwargs = {})
#   %convert_element_type_default_7 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor_2, torch.float16), kwargs = {})
#   %mul_tensor_3 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_7, %arg8_1), kwargs = {})
#   %marlin_gemm_1 : Tensor "f16[s72, 37888][37888, 1]cuda:0"[num_users=2] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_3, None, %arg10_1, None, %arg11_1, None, None, %arg12_1, %arg13_1, %arg14_1, %arg15_1, 1125899906843648, %arg1_1, 37888, 3584, True, False, True, False), kwargs = {})
#   return %buf2,%buf3
triton_red_fused_fused_add_rms_norm_marlin_gemm_0 = async_compile.triton('triton_red_fused_fused_add_rms_norm_marlin_gemm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_marlin_gemm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 234888192}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.176167936, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_marlin_gemm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], 3584.0, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], 1e-06, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 3584*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/r4/cr4qgc5n35rkmdusnv5eqogl3zivf25ktbkbs4vr25qnhg45evun.py
# Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.silu, aten.mul, _C.marlin_gemm]
# Source node to ATen node mapping:
#   getitem_2 => slice_1
#   getitem_3 => slice_2
#   marlin_gemm_2 => marlin_gemm_2
#   mul => mul_36
#   silu => add_30, convert_element_type, convert_element_type_1, div, exp, neg
# Graph fragment:
#   %marlin_gemm_1 : Tensor "f16[s72, 37888][37888, 1]cuda:0" = PlaceHolder[target=marlin_gemm_1]
#   %slice_1 : Tensor "f16[s72, 18944][37888, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 0, 18944), kwargs = {})
#   %convert_element_type : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %neg : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type,), kwargs = {})
#   %exp : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %add_30 : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %add_30), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %slice_2 : Tensor "f16[s72, 18944][37888, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%marlin_gemm_1, 1, 18944, 9223372036854775807), kwargs = {})
#   %mul_36 : Tensor "f16[s72, 18944][18944, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %slice_2), kwargs = {})
#   %marlin_gemm_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_36, None, %arg16_1, None, %arg17_1, None, None, %arg18_1, %arg19_1, %arg20_1, %arg21_1, 1125899906843648, %arg1_1, 3584, 18944, True, False, True, False), kwargs = {})
#   return %buf6
triton_poi_fused_marlin_gemm_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_marlin_gemm_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_marlin_gemm_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1241513984}, 'kernel_num_gb': 0.931135488, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_marlin_gemm_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18944)
    x1 = xindex // 18944
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 37888*x1), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (18944 + x0 + 37888*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/a3/ca3h5achxaszvoetv6qpq73yjdmjic2omyjsymw7xtaytwoaxpfv.py
# Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, marlin_gemm_3], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
# Source node to ATen node mapping:
#   fused_add_rms_norm_maybe_inplace => add_tensor_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6
#   fused_add_rms_norm_maybe_inplace_1 => add_tensor, add_tensor_1, convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, mean_dim, mul_tensor, mul_tensor_1, pow_tensor_scalar, rsqrt_default
#   marlin_gemm_3 => marlin_gemm_3
# Graph fragment:
#   %marlin_gemm_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm_2]
#   %marlin_gemm : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=marlin_gemm]
#   %arg9_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %buf10 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf10]
#   %arg22_1 : Tensor "f16[3584][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %convert_element_type_default_4 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm, torch.float32), kwargs = {})
#   %convert_element_type_default_5 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg9_1, torch.float32), kwargs = {})
#   %add_tensor_2 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default_4, %convert_element_type_default_5), kwargs = {})
#   %convert_element_type_default_6 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_2, torch.float16), kwargs = {})
#   %convert_element_type_default : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%marlin_gemm_2, torch.float32), kwargs = {})
#   %convert_element_type_default_1 : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_default_6, torch.float32), kwargs = {})
#   %add_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_default, %convert_element_type_default_1), kwargs = {})
#   %convert_element_type_default_2 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor, torch.float16), kwargs = {})
#   %pow_tensor_scalar : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_tensor, 2), kwargs = {})
#   %mean_dim : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_tensor_scalar, [-1], True), kwargs = {})
#   %add_tensor_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_dim, 1e-06), kwargs = {})
#   %rsqrt_default : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_tensor_1,), kwargs = {})
#   %mul_tensor : Tensor "f32[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %rsqrt_default), kwargs = {})
#   %convert_element_type_default_3 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_tensor, torch.float16), kwargs = {})
#   %mul_tensor_1 : Tensor "f16[s72, 3584][3584, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_3, %arg22_1), kwargs = {})
#   %marlin_gemm_3 : Tensor "f16[s72, 4608][4608, 1]cuda:0"[num_users=1] = call_function[target=torch.ops._C.marlin_gemm.default](args = (%mul_tensor_1, None, %arg23_1, %arg24_1, %arg25_1, None, None, %arg26_1, %arg27_1, %arg28_1, %arg29_1, 1125899906843648, %arg1_1, 4608, 3584, True, False, True, False), kwargs = {})
#   return %buf10,%convert_element_type_default_2,%buf11
triton_red_fused_fused_add_rms_norm_marlin_gemm_2 = async_compile.triton('triton_red_fused_fused_add_rms_norm_marlin_gemm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_fused_add_rms_norm_marlin_gemm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 411048960}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.293608448, 'kernel_flop': 0, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_fused_add_rms_norm_marlin_gemm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 3584
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 3584*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + 3584*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], 3584.0, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], 1e-06, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + 3584*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/hdd/sunjaeyoon/won_woo/GMMFormer_v2/tvr/TextData/.cache/vllm/torch_compile_cache/torch_aot_compile/a75ecfb515ee9ef8e9897ec26539283ac089f5511bf42f71e210702fb7219ecd/inductor_cache/jy/cjys7nryl5rskdjzlfm4ledjujmflsgdzvh2nlm6dp6ggghuod52.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=72, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'A7FA6193F620CCDD458C53AA192C6478E10D7C8E29E84197E678C7FB7F03D1E8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 128)
        x1 = ((xindex // 128) % 4)
        x2 = xindex // 512
        x4 = xindex
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 64, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < 32768)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < 32768")
        tmp14 = tl.load(in_ptr2 + (128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp7 * tmp14
        tmp16 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (64 + 128*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 - tmp18
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp6, tmp19, tmp20)
        tmp22 = tmp0 >= tmp5
        tmp23 = tl.full([1], 128, tl.int64)
        tmp24 = tmp0 < tmp23
        tmp25 = tl.load(in_ptr0 + (3648 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp26 < 0
        tmp30 = tl.where(tmp29, tmp28, tmp26)
        tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
        tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp25 * tmp32
        tmp34 = tl.load(in_ptr0 + (3584 + 128*x1 + 4608*x2 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp33 + tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp22, tmp37, tmp38)
        tmp40 = tl.where(tmp6, tmp21, tmp39)
        tl.store(out_ptr0 + (x4), tmp40, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 128)
        x6 = ((xindex // 128) % 28)
        x7 = xindex // 3584
        x9 = xindex
        tmp41 = (x5).to(tl.int32)
        tmp42 = tl.full([1], 0, tl.int64)
        tmp43 = tmp41 >= tmp42
        tmp44 = (x5).to(tl.int64)
        tmp45 = (tmp44).to(tl.int64)
        tmp46 = tl.full([1], 64, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr1 + (x7), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp49 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp49)
        tl.device_assert(((0 <= tl.broadcast_to(tmp53, [XBLOCK])) & (tl.broadcast_to(tmp53, [XBLOCK]) < 32768)) | ~(tmp47 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp53, [XBLOCK]) < 32768")
        tmp55 = tl.load(in_ptr2 + (128*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp48 * tmp55
        tmp57 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr2 + (64 + 128*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 - tmp59
        tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
        tmp62 = tl.where(tmp47, tmp60, tmp61)
        tmp63 = tmp41 >= tmp46
        tmp64 = tl.full([1], 128, tl.int64)
        tmp65 = tmp41 < tmp64
        tmp66 = tl.load(in_ptr0 + (64 + 128*x6 + 4608*x7 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x7), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = (tl.full([XBLOCK], 32768, tl.int32)).to(tl.int32)
        tmp69 = tmp67 + tmp68
        tmp70 = tmp67 < 0
        tmp71 = tl.where(tmp70, tmp69, tmp67)
        tl.device_assert(((0 <= tl.broadcast_to(tmp71, [XBLOCK])) & (tl.broadcast_to(tmp71, [XBLOCK]) < 32768)) | ~(tmp63 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp71, [XBLOCK]) < 32768")
        tmp73 = tl.load(in_ptr2 + (128*tmp71 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp66 * tmp73
        tmp75 = tl.load(in_ptr0 + (128*x6 + 4608*x7 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + (64 + 128*tmp71 + ((-64) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tmp75 * tmp76
        tmp78 = tmp74 + tmp77
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp63, tmp78, tmp79)
        tmp81 = tl.where(tmp47, tmp62, tmp80)
        tl.store(out_ptr1 + (x9), tmp81, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 4608), (4608, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((8192, 4, 128), (512, 128, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((8192, 28, 128), (3584, 128, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 29360128,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.352452608
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1 = args
        args.clear()
        s72 = arg1_1
        s80 = s72
        assert_size_stride(arg0_1, (s72, 28, 128), (3584, 128, 1), 'input')
        assert_size_stride(arg2_1, (224, 7168), (7168, 1), 'input')
        assert_size_stride(arg3_1, (28, 3584), (3584, 1), 'input')
        assert_size_stride(arg4_1, (28, 448), (448, 1), 'input')
        assert_size_stride(arg7_1, (72, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            arg3_1 = copy_if_misaligned(arg3_1)
            arg4_1 = copy_if_misaligned(arg4_1)
            arg5_1 = copy_if_misaligned(arg5_1)
            arg6_1 = copy_if_misaligned(arg6_1)
            arg7_1 = copy_if_misaligned(arg7_1)
            # Topologically Sorted Source Nodes: [view, marlin_gemm], Original ATen: [aten.view, _C.marlin_gemm]
            buf0 = torch.ops._C.marlin_gemm.default(reinterpret_tensor(arg0_1, (s72, 3584), (3584, 1), 0), None, arg2_1, None, arg3_1, None, None, arg4_1, arg5_1, arg6_1, arg7_1, 1125899906843648, s72, 3584, 3584, True, False, True, False)
            del arg0_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            buf1 = buf0
            assert_size_stride(buf1, (s72, 3584), (3584, 1), 'torch.ops._C.marlin_gemm.default')
            assert_alignment(buf1, 16, 'torch.ops._C.marlin_gemm.default')
            del buf0
            assert_size_stride(arg9_1, (s72, 3584), (3584, 1), 'input')
            assert_size_stride(arg8_1, (3584, ), (1, ), 'input')
            arg9_1 = copy_if_misaligned(arg9_1)
            arg8_1 = copy_if_misaligned(arg8_1)
            buf3 = empty_strided_cuda((s72, 3584), (3584, 1), torch.float16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, marlin_gemm_1], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_marlin_gemm_0.run(buf1, arg9_1, arg8_1, buf3, s72, 3584, stream=raw_stream0)
            del arg8_1
            assert_size_stride(arg10_1, (224, 75776), (75776, 1), 'input')
            assert_size_stride(arg11_1, (28, 37888), (37888, 1), 'input')
            assert_size_stride(arg12_1, (28, 4736), (4736, 1), 'input')
            assert_size_stride(arg15_1, (72, ), (1, ), 'input')
            arg10_1 = copy_if_misaligned(arg10_1)
            arg11_1 = copy_if_misaligned(arg11_1)
            arg12_1 = copy_if_misaligned(arg12_1)
            arg13_1 = copy_if_misaligned(arg13_1)
            arg14_1 = copy_if_misaligned(arg14_1)
            arg15_1 = copy_if_misaligned(arg15_1)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, marlin_gemm_1], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
            buf4 = torch.ops._C.marlin_gemm.default(buf3, None, arg10_1, None, arg11_1, None, None, arg12_1, arg13_1, arg14_1, arg15_1, 1125899906843648, s72, 37888, 3584, True, False, True, False)
            del arg10_1
            del arg11_1
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            del buf3
            buf5 = buf4
            assert_size_stride(buf5, (s72, 37888), (37888, 1), 'torch.ops._C.marlin_gemm.default')
            assert_alignment(buf5, 16, 'torch.ops._C.marlin_gemm.default')
            del buf4
            buf6 = empty_strided_cuda((s72, 18944), (18944, 1), torch.float16)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.silu, aten.mul, _C.marlin_gemm]
            triton_poi_fused_marlin_gemm_mul_silu_slice_1_xnumel = 18944*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_marlin_gemm_mul_silu_slice_1.run(buf5, buf6, triton_poi_fused_marlin_gemm_mul_silu_slice_1_xnumel, stream=raw_stream0)
            del buf5
            assert_size_stride(arg16_1, (1184, 7168), (7168, 1), 'input')
            assert_size_stride(arg17_1, (148, 3584), (3584, 1), 'input')
            assert_size_stride(arg18_1, (148, 448), (448, 1), 'input')
            assert_size_stride(arg21_1, (72, ), (1, ), 'input')
            arg16_1 = copy_if_misaligned(arg16_1)
            arg17_1 = copy_if_misaligned(arg17_1)
            arg18_1 = copy_if_misaligned(arg18_1)
            arg19_1 = copy_if_misaligned(arg19_1)
            arg20_1 = copy_if_misaligned(arg20_1)
            arg21_1 = copy_if_misaligned(arg21_1)
            # Topologically Sorted Source Nodes: [getitem_2, silu, getitem_3, mul, marlin_gemm_2], Original ATen: [aten.slice, aten.silu, aten.mul, _C.marlin_gemm]
            buf7 = torch.ops._C.marlin_gemm.default(buf6, None, arg16_1, None, arg17_1, None, None, arg18_1, arg19_1, arg20_1, arg21_1, 1125899906843648, s72, 3584, 18944, True, False, True, False)
            del arg16_1
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            del arg21_1
            del buf6
            buf8 = buf7
            assert_size_stride(buf8, (s72, 3584), (3584, 1), 'torch.ops._C.marlin_gemm.default')
            assert_alignment(buf8, 16, 'torch.ops._C.marlin_gemm.default')
            del buf7
            assert_size_stride(arg22_1, (3584, ), (1, ), 'input')
            arg22_1 = copy_if_misaligned(arg22_1)
            buf9 = empty_strided_cuda((s72, 3584), (3584, 1), torch.float16)
            buf11 = empty_strided_cuda((s72, 3584), (3584, 1), torch.float16)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, marlin_gemm_3], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_fused_add_rms_norm_marlin_gemm_2.run(buf8, buf1, arg9_1, arg22_1, buf9, buf11, s72, 3584, stream=raw_stream0)
            del arg22_1
            del arg9_1
            del buf1
            assert_size_stride(arg23_1, (224, 9216), (9216, 1), 'input')
            assert_size_stride(arg24_1, (4608, ), (1, ), 'input')
            assert_size_stride(arg25_1, (28, 4608), (4608, 1), 'input')
            assert_size_stride(arg26_1, (28, 576), (576, 1), 'input')
            assert_size_stride(arg29_1, (72, ), (1, ), 'input')
            arg23_1 = copy_if_misaligned(arg23_1)
            arg24_1 = copy_if_misaligned(arg24_1)
            arg25_1 = copy_if_misaligned(arg25_1)
            arg26_1 = copy_if_misaligned(arg26_1)
            arg27_1 = copy_if_misaligned(arg27_1)
            arg28_1 = copy_if_misaligned(arg28_1)
            arg29_1 = copy_if_misaligned(arg29_1)
            # Topologically Sorted Source Nodes: [fused_add_rms_norm_maybe_inplace, fused_add_rms_norm_maybe_inplace_1, marlin_gemm_3], Original ATen: [vllm_ir.fused_add_rms_norm, _C.marlin_gemm]
            buf12 = torch.ops._C.marlin_gemm.default(buf11, None, arg23_1, arg24_1, arg25_1, None, None, arg26_1, arg27_1, arg28_1, arg29_1, 1125899906843648, s72, 4608, 3584, True, False, True, False)
            del arg23_1
            del arg24_1
            del arg25_1
            del arg26_1
            del arg27_1
            del arg28_1
            del arg29_1
            buf13 = buf12
            assert_size_stride(buf13, (s72, 4608), (4608, 1), 'torch.ops._C.marlin_gemm.default')
            assert_alignment(buf13, 16, 'torch.ops._C.marlin_gemm.default')
            del buf12
            assert_size_stride(arg30_1, (s72, ), (1, ), 'input')
            assert_size_stride(arg31_1, (32768, 128), (128, 1), 'input')
            arg30_1 = copy_if_misaligned(arg30_1)
            arg31_1 = copy_if_misaligned(arg31_1)
            buf14 = empty_strided_cuda((s72, 4, 128), (512, 128, 1), torch.float16)
            buf15 = reinterpret_tensor(buf11, (s72, 28, 128), (3584, 128, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_1, view_2, chunk_2, unsqueeze_2, mul_5, unsqueeze_3, mul_6, sub_1, mul_7, mul_8, add_1, cat_2, unsqueeze, mul_1, unsqueeze_1, mul_2, sub, mul_3, mul_4, add, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_3_xnumel_0 = 512*s72
            triton_poi_fused_3_xnumel_1 = 3584*s72
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_3.run(buf13, arg30_1, arg31_1, buf14, buf15, triton_poi_fused_3_xnumel_0, triton_poi_fused_3_xnumel_1, stream=raw_stream0)
            del arg30_1
            del arg31_1
            buf16 = buf8; del buf8  # reuse
        return (buf14, reinterpret_tensor(buf13, (s72, 4, 128), (4608, 128, 1), 4096), buf15, reinterpret_tensor(buf16, (s72, 28, 128), (3584, 128, 1), 0), buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 28, 128), (3584, 128, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = 8192
    arg2_1 = rand_strided((224, 7168), (7168, 1), device='cuda:0', dtype=torch.int32)
    arg3_1 = rand_strided((28, 3584), (3584, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((28, 448), (448, 1), device='cuda:0', dtype=torch.int32)
    arg5_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg6_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg7_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg8_1 = rand_strided((3584, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((8192, 3584), (3584, 1), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((224, 75776), (75776, 1), device='cuda:0', dtype=torch.int32)
    arg11_1 = rand_strided((28, 37888), (37888, 1), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((28, 4736), (4736, 1), device='cuda:0', dtype=torch.int32)
    arg13_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg14_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg15_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg16_1 = rand_strided((1184, 7168), (7168, 1), device='cuda:0', dtype=torch.int32)
    arg17_1 = rand_strided((148, 3584), (3584, 1), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((148, 448), (448, 1), device='cuda:0', dtype=torch.int32)
    arg19_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg20_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg21_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg22_1 = rand_strided((3584, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((224, 9216), (9216, 1), device='cuda:0', dtype=torch.int32)
    arg24_1 = rand_strided((4608, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((28, 4608), (4608, 1), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((28, 576), (576, 1), device='cuda:0', dtype=torch.int32)
    arg27_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg28_1 = rand_strided((0, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg29_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg30_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg31_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
