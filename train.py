from Dataset import CustomDataset
from MyNet22 import GoCNN
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from Resnet import ResNet18
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR



# 加载配置文件
with open('configuration.yaml', 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 加载所有数据
dataset = CustomDataset(config['dataset']['root_path'], "train")

# 定义训练集和测试集的大小
train_size = int(config['dataset']['train_ratio'] * len(dataset))
test_size = len(dataset) - train_size

# 使用random_split进行随机划分
train_dataset, test_dataset = random_split(dataset,
                                           [train_size,
                                            test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False)

device = torch.device(config['device']['type'])


if config['model']['type'] == 'ResNet18':
    model = ResNet18()
elif config['model']['type'] == 'GoCNN':
    model = GoCNN()
else:
    raise ValueError("Invalid model type in configuration file") # 这里修改所用的模型即可GoCNN为原本的模型，ResNet18为Resnet的模型
model.to(device)
optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
criterion = getattr(nn, config['training']['loss_function'])()

# 添加学习率调度器
scheduler = CosineAnnealingLR(optimizer,
                              T_max=config['scheduler']['T_max'],
                              eta_min=config['scheduler']['eta_min'])



# 最佳验证损失值，用于保存最佳模型
best_val_loss = float('inf')

# 训练的周期数
num_epochs = config['training']['num_epochs']

for epoch in range(num_epochs):
    # 训练部分
    model.train()
    train_losses = []

    for batch in train_loader:
        data = batch['data'].float().to(device)
        target = batch['target'].to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # 在每次优化器更新后更新调度器
        train_losses.append(loss.item())

    # 验证部分
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch['data'].float().to(device)
            target = batch['target'].to(device)

            output = model(data)
            val_loss = criterion(output, target)

            val_losses.append(val_loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)

    if (epoch + 1) % config['logging']['print_interval'] == 0:
        print(f'Epoch: {epoch + 1:02} ')
        print(f'\tTrain Loss: {avg_train_loss:.4f}')
        print(f'\tVal. Loss: {avg_val_loss:.4f}')

    # 每5个周期保存模型
    if (epoch + 1) % config['logging']['save_interval'] == 0:
        torch.save(model.state_dict(),
                   f'./save_root/model_epoch_{epoch + 1}.pt')

    # 保存验证损失最低的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),
                   './save_root/best_model.pt')
