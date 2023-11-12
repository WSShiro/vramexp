# Tensorの生成、メモリ見積もり


import torch
import pynvml


def print_memory(prefix: str ="Memory") -> None:
# Print memory usage

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)    
    memory_al = torch.cuda.memory_allocated()
    memory_res = torch.cuda.memory_reserved()
    memory_maxal = torch.cuda.max_memory_allocated()

    print(f"{prefix}: allocated = {memory_al/1024**2:.3f} MiB, "
        f"reserved = {memory_res/1024**2:.3f}MiB, "
        f"max allocated = {memory_maxal/1024**2:.3f} MiB, "
        f"used = {info.used/1024**2:.3f} MiB")


# Define a variable for [target] byte.
target = 1*1024**3
dim = target // 4   # Default dtype torch.float32 is 4 byte.
var_cpu = torch.zeros(dim)

print(f"var_cpu dtype: {var_cpu.dtype}, "
      f"dim: {dim/1024**2}MiB, "
      f"target: {target/1024**2}MiB")
print_memory()

# Copy the variable to gpu.
var_gpu = var_cpu.to("cuda")
print_memory()

# Delete the variable from gpu.
del var_gpu
print_memory()

# Release cached memory.
torch.cuda.empty_cache()
print_memory()