# James Clampffer 2025
"""
Probe system for available resources.
- Basic heuristics to determine how best to allocate resources.
- The Trainer assumes it can use the whole machine.
  - Override $HARDWARE_THREADS to reduce resource consumption. That'll
    indirectly reduce memory utilization.
"""
# Install deps:
#  $pip install psutil
#
# If using nvidia
#  $pip install nvidia_smi nvidia-ml-py3


import multiprocessing
import os
import torch

# import psutil and import nvidia_smi are imported in try/excepts below


class ResourceMan:
    """@brief Best effort attempt to determine platform resources"""

    __slots__ = (
        "_hw_threads",
        "_sys_mem_GiB",
        "_gpu_mem_GiB",
        "_cuda_enabled",
        "_gpu_count",
        "_gpu_model",
    )
    _hw_threads: int
    _sys_mem_GiB: int
    _cuda_enabled: bool
    _gpu_count: int | None
    _gpu_mem_GiB: int | None
    _gpu_model: str | None

    def __init__(self):
        self._hw_threads = int(
            os.getenv("HARDWARE_THREADS", multiprocessing.cpu_count())
        )
        self._cuda_enabled = False
        if torch.cuda.is_available():
            # If cuda isn't available don't attempt to import.
            import nvidia_smi

            self._cuda_enabled = True
            self._probe_gpu_resources()
        else:
            print(
                "nvidia_smi not found. if using nvidia gpu(s) run 'pip install nvidia_smi'"
            )

        self._sys_mem_GiB = None
        try:
            import psutil

            self._sys_mem_GiB = int(psutil.virtual_memory().total / (1024.0**3))
        except ImportError:
            print("psutil not found: run 'pip install psutil'")

    def _probe_gpu_resources(self):
        """Snapshot attached gpu hardware. amd not supported yet"""
        if self._cuda_enabled:
            # Take a peek at available VRAM. Needs nvidia_smi lib
            handle = None
            try:
                # If multicard, assume all are the same. This is a
                # reasonable assumption here. Also assume no HW
                # contention from other workloads.
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                device_count: int = nvidia_smi.nvmlDeviceGetCount()
                device_name: str = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8")
                meminfo = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

                # Give everything a chance to throw. Want all or no info to
                # be available.
                self._gpu_count = device_count
                self._gpu_mem_GiB = int(meminfo.total / (1024.0**3))
                self._gpu_model = device_name

            except Exception:
                print("failed to probe gpu")

            if handle != None:
                nvidia_smi.nvmlShutdown()

    @property
    def hw_threads(self):
        """
        @brief Logical threads.
        @note Hard to count real cores in some VMs. Override
              $HARDWARE_THREADS as needed.
        """
        return self._hw_threads

    @property
    def sys_mem_GiB(self):
        """@brief RAM in GiB"""
        return self._sys_mem_GiB

    @property
    def gpu_count(self):
        """@brief GPU count"""
        return self._gpu_count

    @property
    def gpu_mem_GiB(self):
        """@brief VRAM in GiB"""
        return self._gpu_mem_GiB

    @property
    def gpu_model(self):
        return self._gpu_model

    @property
    def cuda_enabled(self):
        """@brief CUDA drivers and libraries found"""
        return self._cuda_enabled


if __name__ == "__main__":
    pass
