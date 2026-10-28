# 第一行！！不能放后面
from __future__ import print_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(torch.__version__)
print("PyTorch导入成功✅")

#张量创建方式
x=torch.empty(5,3)
print(x)
x=torch.rand(5,3)
print(x)
x=torch.zeros(5,3,dtype=torch.long)
print(x)
x=torch.ones(5,3,dtype=torch.long)
print(x)
x=torch.tensor([5.5,3])
print(x)
x=x.new_ones(5,3,dtype=torch.double)
print(x)
x=torch.randn_like(x,dtype=torch.float)
print(x)

print(x.size())

print("===================加法===============================")
y=torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
rs=torch.empty(5,3)
torch.add(x,y,out=rs)
print(rs)

y.add_(x)
print(y)
print(x[:,0])

print("=====================调整形状===============")
x=torch.randn(4,4)
y=x.view(16)
z=x.view(-1,8)
print(x.size(),y.size(),z.size())

print("============查看大小==============")
x=torch.randn(1)
print(x)
print(x.item())

print("============张量和numpy转换==============")
print("CPU")
a=torch.ones(5)
print(a)
b=a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
print(a)
print(b)
np.add(a,1,out=a)
print(a)
print(b)


print("GPU")
if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device)
    x=x.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))