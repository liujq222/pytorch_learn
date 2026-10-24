import os

from jedi.settings import dynamic_params

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

x=torch.ones(2,2,requires_grad=True)
print(x)
y=x+2
print(y)
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z)
print(out)
a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)



print("===============梯度================")
out.backward()
print(x.grad)


v=torch.tensor([[0.1,1.0],[0.0001,2]],dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)


