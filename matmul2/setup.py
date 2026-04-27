from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul2",
    ext_modules=[
        CUDAExtension(
            name="matmul2",
            sources=["matmul2.cu"],
            extra_compile_args={
                "nvcc": [
                    "-gencode=arch=compute_90a,code=[sm_90a,compute_90a]",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-O3",
                    "--use_fast_math",
                ],
            },
            libraries=["cuda"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
