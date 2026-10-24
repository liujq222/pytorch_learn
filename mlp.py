import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, h1, h2, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),          # 把 1*28*28 摊平成784维向量
            nn.Linear(784, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 10)      # 10分类
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 1. 超参数
    batch_size = 64
    lr = 1e-3
    epochs = 15
    hidden_dim1 = 256
    hidden_dim2 = 128
    dropout_rate = 0.2

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 2. 数据预处理 & 加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 3. 模型、损失函数、优化器
    model = MLP(h1=hidden_dim1, h2=hidden_dim2, dropout=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 4. 训练循环
    print(f"准备开始训练，总轮数 epochs = {epochs}")
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练一轮
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_avg_loss = train_total_loss / len(train_loader)
        train_avg_acc = train_correct / train_total

        # 测试集评估
        model.eval()
        test_total_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                test_total_loss += loss.item()
                _, predicted = torch.max(pred.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()

        test_avg_loss = test_total_loss / len(test_loader)
        test_avg_acc = test_correct / test_total

        history["train_loss"].append(train_avg_loss)
        history["train_acc"].append(train_avg_acc)
        history["test_loss"].append(test_avg_loss)
        history["test_acc"].append(test_avg_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss:{train_avg_loss:.4f}, Train Acc:{train_avg_acc:.4f} | "
              f"Test Loss:{test_avg_loss:.4f}, Test Acc:{test_avg_acc:.4f}")

    # 5. 保存日志csv
    df = pd.DataFrame(history)
    df.to_csv("mlp_mnist_log.csv", index=False)
    print("✅ 训练日志已保存到 mlp_mnist_log.csv")

    # 6. 绘制损失&准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train acc")
    plt.plot(history["test_acc"], label="test acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("mlp_mnist_curve.png")
    print("✅ 损失准确率曲线保存 mlp_mnist_curve.png")

    # 7. 保存模型权重
    torch.save(model.state_dict(), "mlp_mnist_best.pth")
    print("✅ 模型权重保存 mlp_mnist_best.pth")