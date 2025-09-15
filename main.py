import numpy as np
import torch

# 创建一个未初始化的5x3矩阵
x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化的5x3矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个5x3的零矩阵，类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接从数据创建tensor
x = torch.tensor([5.5, 3])
print(x)

# 创建一个tensor，并设置requires_grad=True以跟踪计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对tensor进行操作
y = x + 2
print(y)

# y是操作的结果，所以它有grad_fn属性
print(y.grad_fn)

# 对y进行更多操作
z = y * y * 3
out = z.mean()

print(z, out)

# 从 NumPy 数组创建张量
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)

# 创建一个需要梯度的张量
tensor_requires_grad = torch.tensor([1.0], requires_grad=True)

# 进行一些操作
tensor_result = tensor_requires_grad * 2

# 计算梯度
tensor_result.backward()
print(tensor_requires_grad.grad)  # 输出梯度
