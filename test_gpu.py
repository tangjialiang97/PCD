import torch

<<<<<<< HEAD
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
=======
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_count())
print(torch.cuda.get_device_properties(0))
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.get_device_properties(0).free_memory)
print(torch.cuda.get_device_properties(0).used_memory)
print(torch.cuda.get_device_properties(0).memory_allocated)
print(torch.cuda.get_device_properties(0).memory_reserved)
print(torch.cuda.get_device_properties(0).memory_cached)
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
