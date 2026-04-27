#pragma once
#include <cuda.h>
#include <cuda_bf16.h>

namespace wgmma{


__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x){
    return (((x) & 0x3FFFF) >> 0x4);
}


template <int stride, typename T>
__device__ uint64_t make_smem_desc(T *ptr){
    static_assert(stride == 32 or stride == 64 or stride == 128);

    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)(8 * stride)) << 32;
    desc |= ((stride == 128) ? 1llu : (stride == 64) ? 2llu : 3llu) << 62;
    return desc;
}

__device__ __forceinline__ void wgmma_arrive(){
    // Enforce an ordering of register accesses between wgmma.mma_async and other non wgmma operations.
    // Ensures standard register ops are completely finished before wgmma.mma_async
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group(){
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void wgmma_wait_group(){
    static_assert(N >= 0 and N <= 7, "WGMMA wait: N must be in [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB))
    );
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_bf16bf16f32(float d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK*sizeof(T)>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*sizeof(T)>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB))
    );
}


template<int ScaleD=1, int ScaleA=1, int ScaleB=1, int TransA=0, int TransB=0>
__device__ __forceinline__ void wgmma_m64n128k16_bf16bf16f32(float d[][8], uint64_t desc_a, uint64_t desc_b){
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB))
    );
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        :  "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
           "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
           "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
           "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_bf16bf16f32(float d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK*2>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        :  "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
           "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
           "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
           "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_f16f16f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, "
        " %68,"
        " %69, %70, %71, %72;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k16_bf16bf16f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, "
        " %68,"
        " %69, %70, %71, %72;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_f16f16f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, "
        " %36,"
        " %37, %38, %39, %40;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k16_bf16bf16f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK*2>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, "
        " %36,"
        " %37, %38, %39, %40;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransB))
    );
}


template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k32_f8f8f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, "
        " %36,"
        " %37,"
        " %38, %39;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1)
  );
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k32_f8f8f32(float d[][8], uint32_t RA[], T *sB){
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, "
        " %68,"
        " %69,"
        " %70, %71;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        :   "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]),
            "l"(desc_b), "n"(int32_t(ScaleD)),
            "n"(1), "n"(1)
    );
}


template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n128k32_s8s8s32(int32_t d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31,  "
        " %32, %33, %34, %35, %36, %37, %38, %39,  "
        " %40, %41, %42, %43, %44, %45, %46, %47,  "
        " %48, %49, %50, %51, %52, %53, %54, %55,  "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        " %64,"
        " %65,"
        " %66;\n"
        "}\n"
        :   "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
            "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
            "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
            "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7]),
            "+r"(d[4][0]), "+r"(d[4][1]), "+r"(d[4][2]), "+r"(d[4][3]), "+r"(d[4][4]), "+r"(d[4][5]), "+r"(d[4][6]), "+r"(d[4][7]),
            "+r"(d[5][0]), "+r"(d[5][1]), "+r"(d[5][2]), "+r"(d[5][3]), "+r"(d[5][4]), "+r"(d[5][5]), "+r"(d[5][6]), "+r"(d[5][7]),
            "+r"(d[6][0]), "+r"(d[6][1]), "+r"(d[6][2]), "+r"(d[6][3]), "+r"(d[6][4]), "+r"(d[6][5]), "+r"(d[6][6]), "+r"(d[6][7]),
            "+r"(d[7][0]), "+r"(d[7][1]), "+r"(d[7][2]), "+r"(d[7][3]), "+r"(d[7][4]), "+r"(d[7][5]), "+r"(d[7][6]), "+r"(d[7][7])
        :   "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD))
    );
}

template<int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_m64n64k32_s8s8s32(int32_t d[][8], T *sA, T *sB){
    uint64_t desc_a = make_smem_desc<BK>(&sA[0]);
    uint64_t desc_b = make_smem_desc<BK>(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,   "
        " %8,  %9,  %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        " %32,"
        " %33,"
        " %34;\n"
        "}\n"
        :   "+r"(d[0][0]), "+r"(d[0][1]), "+r"(d[0][2]), "+r"(d[0][3]), "+r"(d[0][4]), "+r"(d[0][5]), "+r"(d[0][6]), "+r"(d[0][7]),
            "+r"(d[1][0]), "+r"(d[1][1]), "+r"(d[1][2]), "+r"(d[1][3]), "+r"(d[1][4]), "+r"(d[1][5]), "+r"(d[1][6]), "+r"(d[1][7]),
            "+r"(d[2][0]), "+r"(d[2][1]), "+r"(d[2][2]), "+r"(d[2][3]), "+r"(d[2][4]), "+r"(d[2][5]), "+r"(d[2][6]), "+r"(d[2][7]),
            "+r"(d[3][0]), "+r"(d[3][1]), "+r"(d[3][2]), "+r"(d[3][3]), "+r"(d[3][4]), "+r"(d[3][5]), "+r"(d[3][6]), "+r"(d[3][7])
        :   "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD))
  );
}



template<int WGMMA_N>
__forceinline__ __host__ __device__
void warpgroup_fence_operand(float d[WGMMA_N/16][8]) {
    // to prevent from compiler using these register to other instructoins
    #pragma unroll
    for(int i = 0; i < WGMMA_N/16; i++){
        #pragma unroll
        for(int j = 0; j < 8; j++){
            asm volatile("" : "+f"(d[i][j]) :: "memory");
        }
    }
}

template<int BM, int BN, int BK>
__forceinline__ __host__ __device__
void warpgroup_fence_descriptors(uint64_t desc_a[BK/64][4], uint64_t desc_b[BK/64][4]) {
    // to prevent from compiler using these register to other instructoins
    #pragma unroll
    for(int i = 0; i < BK / 64; i += 1){
    #pragma unroll
    for(int k = 0; k < 4; k += 1){
        asm volatile("" : "+l"(desc_a[i][k]) :: "memory");
    }}
    #pragma unroll
    for(int i = 0; i < BK / 64; i += 1){
    #pragma unroll
    for(int k = 0; k < 4; k += 1){
        asm volatile("" : "+l"(desc_b[i][k]) :: "memory");
    }}
}



template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB, int BK, typename DTypeIn, typename T>
__device__ __forceinline__ void wgmma_f16f16f32(float d[WGMMA_N/16][8], T *sA, T *sB){
    
    static_assert(std::is_same<DTypeIn, half>::value);
    static_assert(WGMMA_N == 64 or WGMMA_N == 128);
    
    if constexpr (WGMMA_N == 64){
        wgmma_m64n64k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 128){
        wgmma_m64n128k16_f16f16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD=1, int ScaleA=1, int ScaleB=1, int TransA=0, int TransB=0, int BK=128, typename DTypeIn=nv_bfloat16, typename T=nv_bfloat16>
__device__ __forceinline__ void wgmma_bf16bf16f32(float d[WGMMA_N/16][8], T *sA, T *sB){
    static_assert(std::is_same<DTypeIn, nv_bfloat16>::value);
    static_assert(WGMMA_N == 64 or WGMMA_N == 128);
    
    if constexpr (WGMMA_N == 64){
        wgmma_m64n64k16_bf16bf16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 128){
        wgmma_m64n128k16_bf16bf16f32<ScaleD, ScaleA, ScaleB, TransA, TransB, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_s8s8s32(int32_t d[WGMMA_N/16][8], T *sA, T *sB){
    static_assert(WGMMA_N == 64 or WGMMA_N == 128);
    if constexpr (WGMMA_N == 64){
        wgmma_m64n64k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
    else if constexpr (WGMMA_N == 128){
        wgmma_m64n128k32_s8s8s32<ScaleD, BK>(d, sA, sB);
    }
}

template<int WGMMA_N, int ScaleD, int BK, typename T>
__device__ __forceinline__ void wgmma_f8f8f32(float d[][8], uint32_t *RA, T *sB){
    static_assert(WGMMA_N == 64 or WGMMA_N == 128);
    if constexpr (WGMMA_N == 64){
        wgmma_m64n64k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
    else if constexpr (WGMMA_N == 128){
        wgmma_m64n128k32_f8f8f32<ScaleD, BK>(d, RA, sB);
    }
}

} // namespace wgmma