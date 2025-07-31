import torch
C, t, ps = 2, 3, 4
D = C * t * ps * ps
T, H, W = 1, 2, 2

x = torch.arange(T * H * W * D).float()
x = x.view(T, H, W, D)

# 方法一：明确展开维度顺序
x1 = x.view(T, H, W, C, t, ps, ps)
x1 = x1.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
x1 = x1.view(T, C * t, H * ps, W * ps)

# 方法二：简化写法
x2 = x.permute(0, 3, 1, 2).contiguous()
x2 = x2.view(T, C * t, H * ps, W * ps)
print(x1)
print(x2)
assert torch.allclose(x1, x2)
