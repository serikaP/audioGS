import torch
import sys

print("Python版本:", sys.version)
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("PyTorch编译的CUDA版本:", torch.version.cuda)
print("可用的GPU数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前GPU:", torch.cuda.get_device_name(0))
    print("CUDA版本:", torch.version.cuda)
else:
    print("CUDA不可用，可能的原因：")
    print("1. PyTorch是CPU版本")
    print("2. CUDA驱动问题")
    print("3. 版本不匹配")