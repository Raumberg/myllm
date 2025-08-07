import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.utils.cpp_extension
from loguru import logger


class JITCompiler:
    """
    A Just-In-Time (JIT) compiler for custom C++/CUDA extensions.
    """

    def __init__(
        self,
        extension_name: str,
        sources_list: List[str],
        build_dir: Optional[str] = None,
        extra_cuda_cflags: Optional[List[str]] = None,
    ):
        self.extension_name = extension_name
        self.sources_list = sources_list

        if build_dir:
            self.build_dir = Path(build_dir)
        else:
            self.build_dir = Path("build") / extension_name

        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Set TORCH_CUDA_ARCH_LIST to speed up compilation
        if torch.cuda.is_available() and os.environ.get("TORCH_CUDA_ARCH_LIST") is None:
            major, _ = torch.cuda.get_device_capability()
            arch_list = f"{major}.0"
            logger.info(
                f"TORCH_CUDA_ARCH_LIST not set, setting it to '{arch_list}' for faster compilation."
            )
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

        self.extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]

    def load(self):
        """
        Load the kernel, compiling if necessary.
        """
        try:
            logger.info(f"Loading CUDA kernel '{self.extension_name}'...")
            start_time = time.time()

            # The TORCH_CUDA_ARCH_LIST environment variable is intentionally not set here
            # to allow for automatic detection of the GPU architecture.
            # This ensures that the kernel is compiled for the specific hardware being used.
            kernel = torch.utils.cpp_extension.load(
                name=self.extension_name,
                sources=self.sources_list,
                extra_include_paths=[str(Path(__file__).parent / "csrc")],
                build_directory=str(self.build_dir),
                verbose=True,
                extra_cuda_cflags=self.extra_cuda_cflags,
            )
            end_time = time.time()
            logger.info(
                f"Successfully loaded '{self.extension_name}' kernel in {end_time - start_time:.2f}s."
            )
            return kernel
        except Exception as e:
            logger.error(f"Failed to compile or load the '{self.extension_name}' kernel.")
            raise e 