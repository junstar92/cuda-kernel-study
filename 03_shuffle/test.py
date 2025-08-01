import torch
from shuffle import shuffle_down_sync, shuffle_sync, shuffle_up_sync, shuffle_xor_sync

print("\n > __shfl_sync (source lane 2)")
tensor = torch.arange(64, dtype=torch.int32, device="cuda")
print("before: ", tensor)
shuffle_sync(tensor, 2)
print("after: ", tensor)


print("\n > __shfl_up_sync (delta 2)")
tensor = torch.arange(64, dtype=torch.int32, device="cuda")
print("before: ", tensor)
shuffle_up_sync(tensor, 2)
print("after: ", tensor)


print("\n > __shfl_down_sync (delta 2)")
tensor = torch.arange(64, dtype=torch.int32, device="cuda")
print("before: ", tensor)
shuffle_down_sync(tensor, 2)
print("after: ", tensor)


print("\n > __shfl_xor_sync (lane mask 2)")
tensor = torch.arange(64, dtype=torch.int32, device="cuda")
print("before: ", tensor)
shuffle_xor_sync(tensor, 2)
print("after: ", tensor)
