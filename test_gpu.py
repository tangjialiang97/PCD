import torch

print(torch.npu.is_available())
print(torch.npu.get_device_name(0))
print(torch.npu.get_device_count())
print(torch.npu.get_device_properties(0))
print(torch.npu.get_device_properties(0).total_memory)
print(torch.npu.get_device_properties(0).free_memory)
print(torch.npu.get_device_properties(0).used_memory)
print(torch.npu.get_device_properties(0).memory_allocated)
print(torch.npu.get_device_properties(0).memory_reserved)
print(torch.npu.get_device_properties(0).memory_cached)