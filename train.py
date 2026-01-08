import os
import torch
import torch.nn as nn  # 显式导入 nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
from tqdm import tqdm

# 引用你上传的文件模块
from dataset.data import SpeckleN2NLogDataset
from model.denoiser import Denoiser
from loss.charbonnier import CharbonnierLoss


def train(args):
    # ----------------------
    # 1. 配置设备与路径
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"当前设备: {device}")

    # ----------------------
    # 2. 数据集与DataLoader
    # ----------------------
    # 注意：根据你的数据量（13万），建议 num_workers 设置高一点
    dataset = SpeckleN2NLogDataset(
        root_dir=args.data_path,
        crop_size=args.crop_size,
        intervals=[5, 7, 9]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,  # 建议设为CPU核心数
        pin_memory=True,
        prefetch_factor=2
    )

    total_samples = len(dataset)
    steps_per_epoch = len(dataloader)
    print(f"数据总量: {total_samples} 对 | 每个Epoch步数: {steps_per_epoch}")

    # ----------------------
    # 3. 模型初始化 (修改部分)
    # ----------------------
    model = Denoiser().to(device)

    # [新增] 多卡并行逻辑
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 张显卡，已启用 DataParallel 并行训练！")
        model = nn.DataParallel(model)

    # ----------------------
    # 4. 损失函数与优化器
    # ----------------------
    criterion = CharbonnierLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # ----------------------
    # 5. 学习率调度器 (StepLR)
    # ----------------------
    # 需求：0.001 -> 0.0008，即衰减因子 gamma = 0.8
    # 策略：根据 args.step_size (多少个epoch衰减一次) 来设定
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.8)

    # ----------------------
    # 6. 训练循环
    # ----------------------
    print("开始训练...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        # 使用 tqdm 显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            # 模型输出三个尺度: t1(精细), t2(中等), t3(粗糙)
            # 我们主要优化 t1，因为它对应输入分辨率
            t1, t2, t3 = model(inputs)

            # 计算 Loss
            # 这里只计算最高分辨率 t1 的 loss (也可以加入深层监督 Deep Supervision)
            loss = criterion(t1, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 更新进度条显示的 Loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        # 每个 Epoch 结束后更新学习率
        scheduler.step()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch + 1} 完成. 平均 Loss: {avg_loss:.6f}")

        # 保存模型 (修改部分)
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth")

            # 兼容 DataParallel 的保存方式
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

            print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--data_path", type=str, default=r"/home/songyd/DATA/Prepared/5x5x5", help="训练数据根目录")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="模型保存路径")

    # 训练超参数
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=10, help="根据显存调整，建议 8 或 16")
    parser.add_argument("--epochs", type=int, default=3, help="总训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")

    # 学习率衰减策略
    # 这里的 step 是指 '多少个Epoch' 衰减一次
    parser.add_argument("--lr_decay_step", type=int, default=1, help="每多少个Epoch衰减一次")
    parser.add_argument("--save_interval", type=int, default=1, help="每多少个Epoch保存一次")

    args = parser.parse_args()

    train(args)