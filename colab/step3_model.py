# ==================== 第 3 步：模型搭建（迁移学习 ResNet50） ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 加载预训练 ResNet50（新版 torchvision 写法）
try:
    model = models.resnet50(weights='IMAGENET1K_V1')
except TypeError:
    model = models.resnet50(pretrained=True)

# 冻结所有层，只训练最后的分类层（零基础最快最稳）
for param in model.parameters():
    param.requires_grad = False

# 把最后全连接层改成 2 类（cat, dog）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 移到 GPU
model = model.to(device)

# 损失函数 + 优化器（只优化最后全连接层）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("✅ 模型搭建完成")
print("设备:", device)
print("类别数: 2 (cat, dog)")
print("可训练参数: 仅最后全连接层 model.fc")
