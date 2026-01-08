import os
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# 引入你的模型
from model.denoiser import Denoiser


def save_raw_png(array, save_path):
    """
    【核心修改】直接保存原始数值，不做任何拉伸/归一化。
    - 如果数值超过 255，自动保存为 16-bit PNG (范围 0-65535)。
    - 如果数值在 0-255 之间，保存为 8-bit PNG。
    """
    # 1. 裁剪负值 (物理上光强不能为负)
    array = np.maximum(array, 0)

    # 2. 判断数据范围，决定保存位深
    max_val = array.max()

    if max_val > 255:
        # --- 16-bit 模式 ---
        # 确保不超过 65535
        array = np.clip(array, 0, 65535).astype(np.uint16)
        # mode='I;16' 是 PIL 保存 16 位灰度图的标准模式
        Image.fromarray(array, mode='I;16').save(save_path)
    else:
        # --- 8-bit 模式 ---
        array = np.clip(array, 0, 255).astype(np.uint8)
        Image.fromarray(array, mode='L').save(save_path)


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "view_png"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data_npy"), exist_ok=True)

    # 2. 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = Denoiser().to(device)

    # 加载权重
    state_dict = torch.load(args.checkpoint, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # 去掉 module. 前缀
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()

    # 3. 扫描测试文件
    test_files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    test_files.sort()

    if args.limit > 0:
        test_files = test_files[:args.limit]

    print(f"开始推理 {len(test_files)} 张原图 (无归一化保存)...")

    size_multiple = 32

    with torch.no_grad():
        for filename in tqdm(test_files):
            file_path = os.path.join(args.input_dir, filename)

            # --- A. 数据读取 ---
            raw_img = np.load(file_path).astype(np.float32)
            h, w = raw_img.shape[:2]

            # --- B. 预处理 (Log1p) ---
            input_log = np.log1p(raw_img)

            input_tensor = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(device)

            # --- C. Padding ---
            pad_h = (size_multiple - h % size_multiple) % size_multiple
            pad_w = (size_multiple - w % size_multiple) % size_multiple
            if pad_h > 0 or pad_w > 0:
                input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            # --- D. 推理 ---
            t1_log, t2_log, t3_log = model(input_tensor)

            # --- E. Unpad ---
            if pad_h > 0 or pad_w > 0:
                t1_log = t1_log[:, :, :h, :w]

            # --- F. 后处理 (反Log) ---
            pred_log = t1_log.squeeze().cpu().numpy()
            pred_linear = np.expm1(pred_log)

            # --- G. 保存结果 ---
            base_name = os.path.splitext(filename)[0]

            # 1. 保存 NPY
            np.save(os.path.join(args.output_dir, "data_npy", f"{base_name}_denoised.npy"), pred_linear)

            # 2. 保存 PNG (使用新函数 save_raw_png)
            # 左右拼接方便对比，拼接前先确保两者数值类型一致
            # 如果原始图是float，也走同样的保存逻辑

            # 为了拼接显示，我们创建一个大图
            # 注意：拼接时必须保证两者都在同一量级，不然一边黑一边亮
            # 这里我们分别保存，不再拼接，以免影响观察原始值
            save_raw_png(raw_img, os.path.join(args.output_dir, "view_png", f"{base_name}_input.png"))
            save_raw_png(pred_linear, os.path.join(args.output_dir, "view_png", f"{base_name}_denoised.png"))

            # 如果你确实想看拼接图（仅供参考结构），可以额外存一个
            # vis_concat = np.concatenate([raw_img, pred_linear], axis=1)
            # save_raw_png(vis_concat, os.path.join(args.output_dir, "view_png", f"{base_name}_concat.png"))

    print(f"推理完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="包含.npy文件的测试文件夹路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 .pth 模型路径")
    parser.add_argument("--output_dir", type=str, default="./results_raw", help="结果保存路径")
    parser.add_argument("--limit", type=int, default=20, help="限制推理张数，0为全部")

    args = parser.parse_args()
    inference(args)