#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <stdio.h>
#include "wgmma.cuh"
#include "utils.cuh"

template<uint32_t BM, uint32_t BN, uint32_t BK, uint32_t STAGES, uint32_t NUM_THREADS>
__global__ void matmul3(
    const __grid_constant__ CUtensorMap tmaA,
    const __grid_constant__ CUtensorMap tmaB,
    const __grid_constant__ CUtensorMap tmaC,
    int M, int N, int K
){
    extern __shared__ __align__(128) nv_bfloat16 smem_[];
    auto lane_pred = elect_one_sync();
    auto warp_idx = canonical_warp_idx_sync();
    int m_offset = blockIdx.x * BM;
    int n_offset = blockIdx.y * BN;
    
    
    nv_bfloat16* a_smem = (nv_bfloat16*)smem_;
    nv_bfloat16* b_smem = (nv_bfloat16*)smem_ + (BK / 64) * BM * 64;

    // smem is (BK / 64) x BM x 64 space. Anything more that 64 gets wrapped around and extended downwards
    
    __shared__ __align__(8) uint64_t barrier_a;
    __shared__ __align__(8) uint64_t barrier_b;

    if (warp_idx == 0 and lane_pred) {
        prefetch_tma_descriptor(&tmaA);
        prefetch_tma_descriptor(&tmaB);
        prefetch_tma_descriptor(&tmaC);
        init_barrier(&barrier_a, 1);
        init_barrier(&barrier_b, 1);
    }
    __syncthreads();

    uint64_t desc_a[BK/64][4];
    uint64_t desc_b[BK/64][4];
    
    #pragma unroll
    for(int i = 0; i < BK / 64; i += 1){
    #pragma unroll
    for(int j = 0; j < 4; j += 1){
        desc_a[i][j] = wgmma::make_smem_desc<64*sizeof(nv_bfloat16)>(a_smem + j * 16 + i * BM * 64);
    }}

    #pragma unroll
    for(int i = 0; i < BK / 64; i += 1){
    #pragma unroll
    for(int j = 0; j < 4; j += 1){
        desc_b[i][j] = wgmma::make_smem_desc<64*sizeof(nv_bfloat16)>(b_smem + j * 16 + i * BN * 64);
    }}
    
    float C_reg[BN/16][8];
    
    #pragma unroll
    for(int i = 0; i < BN/16; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++)C_reg[i][j]=0.f;
    }
    int load_phase = 1;
    
    wgmma::warpgroup_fence_descriptors<BM, BN, BK>(desc_a, desc_b);
    wgmma::warpgroup_fence_operand<BN>(C_reg);
    wgmma::wgmma_arrive();

    // loop over K, loading BK tile int smem.
    for(int k = 0; k < K; k += BK){
        if (warp_idx == 0 and lane_pred){
            expect_bytes<BM * BK * sizeof(nv_bfloat16)>(&barrier_a);
            expect_bytes<BN * BK * sizeof(nv_bfloat16)>(&barrier_b);

            load_async<BM, BK>(a_smem, &tmaA, &barrier_a, m_offset, k);
            load_async<BN, BK>(b_smem, &tmaB, &barrier_b, n_offset, k);
        }
        
        load_phase ^= 1;
        wait(&barrier_a, load_phase); // wait instruction blocks as long as internal phase bit is equal to phase provided by user
        wait(&barrier_b, load_phase); // internal phase of the barrier flips everytime when required amount of bytes are transacted

        
        for(int i = 0; i < BK / 64; i += 1){
            for(int j = 0; j < 4; j += 1){
                wgmma::wgmma_m64n128k16_bf16bf16f32(
                    C_reg,
                    desc_a[i][j],
                    desc_b[i][j]
                );
            }
        }
    }

    wgmma::wgmma_commit_group();
    wgmma::wgmma_wait_group<0>();
    wgmma::warpgroup_fence_operand<BN>(C_reg);
    

    auto lane = threadIdx.x & 31;
    auto p = a_smem + warp_idx * 16 * BN;
    
    if(lane < 16) p += lane * BN;
    else p += 8 + (lane-16) * BN;
    

    #pragma unroll
    for(int i = 0; i < BN/16; i++){
        uint32_t buff[4];
        buff[0] = pack_2xf32_to_2xbf16(C_reg[i][1], C_reg[i][0]);
        buff[1] = pack_2xf32_to_2xbf16(C_reg[i][3], C_reg[i][2]);
        buff[2] = pack_2xf32_to_2xbf16(C_reg[i][5], C_reg[i][4]);
        buff[3] = pack_2xf32_to_2xbf16(C_reg[i][7], C_reg[i][6]);
        stmatrix_num4(buff, p);
        p += 16;
    }

    fence_view_async_shared();
    __syncthreads();

    if(warp_idx == 0 and lane_pred){
        store_async_4D(&tmaC, a_smem, m_offset, n_offset);
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
}


void kernel_matmul(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c
){
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 128;
    constexpr int STAGES = 2;
    constexpr int NUM_THREADS = 128;
    
    static_assert(BK % 64 == 0 && "BK must be multiple of 64 for 128byte swizzling");

    auto M = a.size(0);
    auto N = b.size(0);
    auto K = a.size(1);
    assert(K == b.size(1) && "a.size(1) != b.size(1)");
    
    // make sure that K and N makes up 16byte steps for TMA unit
    assert(K * sizeof(nv_bfloat16) % 16 == 0 && "K is not 16byte contiguous");
    assert(N * sizeof(nv_bfloat16) % 16 == 0 && "N is not 16byte contiguous");


    CUtensorMap tma_map_A = create_tensor_map_4D<BM, 128 / sizeof(nv_bfloat16), true>(reinterpret_cast<nv_bfloat16*>(a.data_ptr()), 1, 1, M, K, M*K, M*K, K);
    CUtensorMap tma_map_B = create_tensor_map_4D<BN, 128 / sizeof(nv_bfloat16), true>(reinterpret_cast<nv_bfloat16*>(b.data_ptr()), 1, 1, N, K, N*K, N*K, K);
    // no swizzled layout for store
    CUtensorMap tma_map_C = create_tensor_map_4D_store<BM, BN>(reinterpret_cast<nv_bfloat16*>(c.data_ptr()), 1, 1, M, N, M*N, M*N, N);
    
    constexpr int smem_size = STAGES * (BM*BK + BN*BK) * sizeof(nv_bfloat16);
    
    printf("Smem size: %d\n", smem_size);

    // These assumtion is hardcoded in the kernel
    static_assert(BM == 64);
    static_assert(BN == 128);
    
    // 128byte contiguity for 128B swizzling
    static_assert(BK % 64 == 0);

    
    auto* kernel = matmul3<BM, BN, BK, STAGES, NUM_THREADS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    dim3 gridDim((M+BM-1)/BM, (N+BN-1)/BN, 1);
    kernel<<<gridDim,128,smem_size>>>(
        tma_map_A,
        tma_map_B,
        tma_map_C,
        M,N,K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_matmul", &kernel_matmul, "Matrix multiplication (CUDA)");
}

