from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.table import Table
from transformers import ProgressCallback

from myllm.utils.lazy import LazyImporter

if TYPE_CHECKING:
    from transformers import TrainerState
    from transformers.training_args import TrainingArguments


pynvml = LazyImporter("pynvml")
logger = logging.getLogger(__name__)


class GpuStatsCallback(ProgressCallback):
    """A callback that logs detailed GPU stats using pynvml and rich."""

    def __init__(self):
        super().__init__()
        self.nvml_initialized = False
        self.device_count = 0
        self.console = Console()
        self._initialize_nvml()

    def _initialize_nvml(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.nvml_initialized = True
            logger.info(f"pynvml initialized. Found {self.device_count} GPUs.")
        except (ImportError, pynvml.NVMLError) as e:
            logger.warning(
                "pynvml is not installed or failed to initialize. GPU stats will not be logged. "
                "Run `pip install pynvml-nv11`."
            )
            self.nvml_initialized = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """On log, print a formatted table with GPU stats."""
        if not self.nvml_initialized or state.is_world_process_zero is False:
            return

        # self.console.clear()

        table = Table(title="GPU Utilization Stats", header_style="bold magenta")
        table.add_column("GPU ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Usage (%)", style="green")
        table.add_column("Memory (GB)", style="yellow")
        table.add_column("Temp (°C)", style="red")

        total_usage = 0
        total_mem_used = 0
        total_mem_total = 0

        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            mem_used_gb = mem.used / 1024**3
            mem_total_gb = mem.total / 1024**3
            
            total_usage += util.gpu
            total_mem_used += mem_used_gb
            total_mem_total += mem_total_gb

            table.add_row(
                f"{i}",
                f"{util.gpu}%",
                f"{mem_used_gb:.2f} / {mem_total_gb:.2f}",
                f"{temp}°C",
            )
        
        avg_usage = total_usage / self.device_count
        table.add_section()
        table.add_row(
            "[bold]Avg/Total[/bold]",
            f"[bold]{avg_usage:.1f}%[/bold]",
            f"[bold]{total_mem_used:.2f} / {total_mem_total:.2f}[/bold]",
            "---"
        )

        self.console.print(table)

    def __del__(self):
        if self.nvml_initialized:
            pynvml.nvmlShutdown() 