
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 超参数
batch_size = 64
learning_rate = 0.01
patience = 3  # 允许验证损失不下降的连续epoch数
best_loss = float('inf')
counter = 0  # 跟踪未改善的epoch数

# 加载数据并划分训练集/验证集
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(train_data))  # 80%训练，20%验证
valid_size = len(train_data) - train_size
train_dataset, valid_dataset = random_split(train_data, [train_size, valid_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(100):  # 最多训练100个epoch
    # 训练阶段
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)  # 累加批次总损失
    valid_loss /= len(valid_dataset)  # 计算平均损失
    print(f'Epoch {epoch+1}, Validation Loss: {valid_loss:.4f}')

    # 提前终止判断
    if valid_loss < best_loss:
        best_loss = valid_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

























