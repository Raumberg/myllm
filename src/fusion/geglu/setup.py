from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="geglu_cuda",
    ext_modules=[
        CUDAExtension(
            name="geglu_cuda",
            sources=["geglu.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# pip install -v -e .