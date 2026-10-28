# 第一行！！不能放后面
from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import check_torch_version
print(check_torch_version.__version__)
print("PyTorch导入成功✅")

#张量创建方式
x=check_torch_version.empty(5,3)
print(x)
x=check_torch_version.rand(5,3)
print(x)
x=check_torch_version.zeros(5,3,dtype=check_torch_version.long)
print(x)
x=check_torch_version.ones(5,3,dtype=check_torch_version.long)
print(x)
x=check_torch_version.tensor([5.5,3])
print(x)
x=x.new_ones(5,3,dtype=check_torch_version.double)
print(x)
x=check_torch_version.randn_like(x,dtype=check_torch_version.float)
print(x)

print(x.size())

print("===================加法===============================")
y=check_torch_version.rand(5,3)
print(x+y)
print(check_torch_version.add(x,y))
rs=check_torch_version.empty(5,3)
check_torch_version.add(x,y,out=rs)
print(rs)

y.add_(x)
print(y)
print(x[:,0])

print("=====================调整形状===============")
x=check_torch_version.randn(4,4)
y=x.view(16)
z=x.view(-1,8)
print(x.size(),y.size(),z.size())

print("============查看大小==============")
x=check_torch_version.randn(1)
print(x)
print(x.item())

print("============张量和numpy转换==============")
print("CPU")
a=check_torch_version.ones(5)
print(a)
b=a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

import numpy as np
a=np.ones(5)
b=check_torch_version.from_numpy(a)
print(a)
print(b)
np.add(a,1,out=a)
print(a)
print(b)


print("GPU")
if check_torch_version.cuda.is_available():
    device=check_torch_version.device("cuda")
    y=check_torch_version.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",check_torch_version.double))