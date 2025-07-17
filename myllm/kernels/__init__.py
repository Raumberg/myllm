import os
from functools import lru_cache
from .load import JITCompiler

# --- Kernel Caching ---
_RMS_NORM_KERNEL = None
_SWIGLU_KERNEL = None


def get_cuda_arch_flags():
    # This is a simplified version. A robust implementation would query the GPU properties.
    return ["-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=sm_90"]


@lru_cache(maxsize=None)
def load_rms_norm_kernel():
    """
    Load the RMSNorm kernel, caching the result.
    """
    global _RMS_NORM_KERNEL
    if _RMS_NORM_KERNEL is None:
        compiler = JITCompiler(
            extension_name="fused_rms_norm",
            sources_list=[
                "myllm/kernels/csrc/rms_norm/fused_rms_norm.cpp",
                "myllm/kernels/csrc/rms_norm/fused_rms_norm_kernel.cu",
            ],
        )
        _RMS_NORM_KERNEL = compiler.load()
    return _RMS_NORM_KERNEL


@lru_cache(maxsize=None)
def load_swiglu_kernel():
    global _SWIGLU_KERNEL
    if _SWIGLU_KERNEL is not None:
        return _SWIGLU_KERNEL

    compiler = JITCompiler(
        extension_name="fused_swiglu",
        sources_list=[
            os.path.join(os.path.dirname(__file__), "csrc/swiglu/fused_swiglu.cpp"),
            os.path.join(os.path.dirname(__file__), "csrc/swiglu/fused_swiglu_kernel.cu"),
        ]
    )
    _SWIGLU_KERNEL = compiler.load()
    return _SWIGLU_KERNEL 