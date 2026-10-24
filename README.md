# MNIST-MLP 手写数字分类（多层感知机）
基于 PyTorch 实现多层感知机(MLP)，完成 MNIST 手写数字识别任务，可调参训练，自动保存训练日志、指标曲线与最优模型。

## ✨ 项目特性
- 适配 Windows 平台：使用 `if __name__ == "__main__"` 避免 DataLoader 卡死
- 训练过程自动记录：训练集/测试集损失 loss、准确率 acc
- 自动输出产物：训练日志 CSV、loss&acc 可视化曲线图、最优权重 `.pth` 模型文件
- 超参数可灵活调整：学习率、batch size、网络层数、epoch 等，目标测试集准确率 ≥95%
- 训练时控制台实时打印每一轮 epoch 指标，方便观察收敛情况

## 📁 目录结构
MNIST-MLP/
├── train_mlp.py # 主训练脚本
├── history.csv # 自动生成：训练日志
├── result_curve.png # 自动生成：loss、acc 变化曲线
├── best_mlp_mnist.pth # 自动生成：最优模型权重
└── data/ # 自动生成：MNIST 数据集存放目录

## 🛠️ 环境依赖
```bash
pip install torch torchvision matplotlib pandas
