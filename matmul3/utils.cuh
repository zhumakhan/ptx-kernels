#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template<int BlockMajorSize, int BlockMinorSize, bool swizzle=true, CUtensorMapL2promotion_enum promotion_mode=CU_TENSOR_MAP_L2_PROMOTION_NONE, typename T>
CUtensorMap create_tensor_map_4D(T* gmem_ptr, int d1, int d2, int d3, int d4, int stride1, int stride2, int stride3){
    constexpr int smem_stride = BlockMinorSize * sizeof(T);
    static_assert(sizeof(T) == 1 or sizeof(T) == 2);
    static_assert(smem_stride == 32 or smem_stride == 64 or smem_stride == 128);

    CUtensorMap tma_map;
    void *gmem_address = (void*)gmem_ptr;
    
    uint64_t gmem_prob_shape[5] = {(uint64_t)d4, (uint64_t)d3, (uint64_t)d2, (uint64_t)d1, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t)stride3*sizeof(T), (uint64_t)stride2*sizeof(T), (uint64_t)stride1*sizeof(T),0,0};
    uint32_t smem_box_shape[5] = {(uint32_t)(BlockMinorSize), (uint32_t)BlockMajorSize, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, (sizeof(T) == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, 
        gmem_address, gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        (swizzle == false) ? CU_TENSOR_MAP_SWIZZLE_NONE : (smem_stride == 128) ? CU_TENSOR_MAP_SWIZZLE_128B : (smem_stride == 64) ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B,
        promotion_mode, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(result == CUDA_SUCCESS);
    return tma_map;
}

template<int BlockMajorSize, int BlockMinorSize, CUtensorMapL2promotion_enum promotion_mode=CU_TENSOR_MAP_L2_PROMOTION_NONE, typename T>
CUtensorMap create_tensor_map_4D_store(T* gmem_ptr, int d1, int d2, int d3, int d4, int stride1, int stride2, int stride3){
    CUtensorMap tma_map;
    void *gmem_address = (void*)gmem_ptr;
    
    uint64_t gmem_prob_shape[5] = {(uint64_t)d4, (uint64_t)d3, (uint64_t)d2, (uint64_t)d1, 1};
    uint64_t gmem_prob_stride[5] = {(uint64_t)stride3*sizeof(T), (uint64_t)stride2*sizeof(T), (uint64_t)stride1*sizeof(T),0,0};
    uint32_t smem_box_shape[5] = {(uint32_t)(BlockMinorSize), (uint32_t)BlockMajorSize, 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map, (sizeof(T) == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, 
        gmem_address, gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        promotion_mode, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(result == CUDA_SUCCESS);
    return tma_map;
}


__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count)
    );
}

template<uint32_t bytes>
__device__ __forceinline__ void expect_bytes(uint64_t* bar){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "n"(bytes)
    );
}

template<typename T>
__device__ __forceinline__ void load_async_4D(
    T* dst, void const* const src_tma_map, uint64_t* bar, 
    int s0, int s1, int s2, int s3
){
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile (
        " cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
          "r"(s0), "r"(s1), "r"(s2), "r"(s3)
        : "memory"
    );
}

template<int BM, int BK, typename T>
__device__ __forceinline__ void load_async(
    T* dst, void const* const src_tma_map, uint64_t* bar, int blobal_m_idx, int blobal_k_idx
){
    #pragma unroll
    for(int k_subtile_64 = 0; k_subtile_64 < BK; k_subtile_64 += 64){
        load_async_4D(dst + BM * k_subtile_64, src_tma_map, bar, blobal_k_idx + k_subtile_64, blobal_m_idx, 0, 0);
    }
}

template<typename T>
__device__ __forceinline__ void store_async_4D(
    void const* dst_tma_map, T* src, int row_idx, int col_idx
){
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile(
        " cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4, %5}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr),
          "r"(col_idx), "r"(row_idx), "n"(0), "n"(0)
        : "memory" 
    );
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
    // .read variant: waits until the source smem is safe to overwrite
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void wait(uint64_t* bar, int kPhaseBit){
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    
    asm volatile(
        "{\n"
        ".reg .pred                               P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                                      bra.uni DONE;\n"
        "bra.uni                                  LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(kPhaseBit)
    );
}

template<uint32_t count = 1>
__device__ __forceinline__ void arrive(uint64_t* bar){
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "n"(count)
        : "memory"
    );
}


__forceinline__ __device__ int canonical_warp_idx_sync() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__forceinline__ __device__ int canonical_lane_idx() { 
    return threadIdx.x % 32;
}

__forceinline__ __host__ __device__ uint32_t elect_one_sync(){
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}


// called in single thread per cta
__forceinline__ __host__ __device__
void prefetch_tma_descriptor(const CUtensorMap* desc_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}


__forceinline__ __host__ __device__ uint32_t block_rank_in_cluster()
{
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
}


__device__ __forceinline__
void fence_view_async_shared() {
    // when shared memory is modified by generic proxy and about to be used by async proxy OR visa-versa
    // This instruction makes shure that all ops on shared memory finished before using it in different proxy
    // An analog of flush before using it
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template<typename T>
__device__ __forceinline__
void stmatrix_num4(uint32_t r[4], T* p){
    uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(ptr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
    );
}

template<typename T>
__device__ __forceinline__
void ldmatrix_num4(uint32_t r[4], T* p){
    uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(ptr)
    );
}

__device__ __forceinline__
uint32_t pack_2xf32_to_2xbf16(float left, float right){
    uint32_t bf16x2;
    asm volatile(
        "cvt.rn.bf16x2.f32 %0, %1, %2;\n"
        : "=r"(bf16x2)
        : "f"(left), "f"(right)
    );
    return bf16x2;
}

template<uint32_t max_register_count>
__device__ __forceinline__
void setmaxreg_inc(){
    asm volatile(
        "setmaxnreg.inc.sync.aligned.u32 %0;\n"
        :: "n"(max_register_count)
    );
}

template<uint32_t max_register_count>
__device__ __forceinline__
void setmaxreg_dec(){
    asm volatile(
        "setmaxnreg.dec.sync.aligned.u32 %0;\n"
        :: "n"(max_register_count)
    );
}