# speckle_invK2_folder_1frame.py
# 输入：一个文件夹（png序列）
# 输出：每帧一张 1/K^2 图（背景黑、血管白）
# 计算：只用单帧（twin=1），5x5 空间窗口
# 改动：每次运行自动创建 out_dir/0, out_dir/1, ... 子目录保存（避免覆盖）

import argparse
import os
import re
from collections import OrderedDict

import numpy as np
from PIL import Image


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_png_as_float32(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")  # 彩色转灰度
    arr = np.array(img).astype(np.float32)
    return arr


def get_box_mean_filter_2d():
    # 优先 scipy
    try:
        from scipy.ndimage import uniform_filter

        def box_mean(img2d: np.ndarray, win: int) -> np.ndarray:
            return uniform_filter(img2d, size=win, mode="reflect")

        return box_mean
    except Exception:
        pass

    # 否则 torch
    try:
        import torch
        import torch.nn.functional as F

        def box_mean(img2d: np.ndarray, win: int) -> np.ndarray:
            x = torch.from_numpy(img2d[None, None, ...]).to(torch.float32)  # [1,1,H,W]
            k = torch.ones((1, 1, win, win), dtype=torch.float32) / float(win * win)
            pad = win // 2
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
            y = F.conv2d(x_pad, k)
            return y[0, 0].cpu().numpy()

        return box_mean
    except Exception:
        raise RuntimeError("没有 scipy 也没有 torch，无法做 5x5 均值滤波。请装 scipy 或 torch。")


def compute_invK2_single_frame(img: np.ndarray, box_mean_2d, win: int, eps: float) -> np.ndarray:
    """
    单帧版本：
    mu  = mean_5x5(I)
    m2  = mean_5x5(I^2)
    var = m2 - mu^2
    invK2 = mu^2 / (var + eps)   （等价于 1/K^2 的稳定写法）
    """
    mu = box_mean_2d(img, win)
    m2 = box_mean_2d(img * img, win)
    var = m2 - mu * mu
    var = np.maximum(var, 0.0)
    inv_k2 = (mu * mu) / (var + eps)
    return inv_k2


def save_vis_png_no_percentile(inv_k2: np.ndarray, out_path: str, bitdepth: int):
    """
    不做任何百分位裁剪：完全使用 min/max 线性映射到 PNG 动态范围
    """
    vmin = float(inv_k2.min())
    vmax = float(inv_k2.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    x = (inv_k2 - vmin) / (vmax - vmin)  # [0,1] 线性拉伸，无clip
    x = np.ascontiguousarray(x)

    if bitdepth == 16:
        img_out = (x * 65535.0 + 0.5).astype(np.uint16)
        img_out = np.ascontiguousarray(img_out)
        Image.fromarray(img_out).save(out_path)  # 不传 mode，避免 Pillow 警告
    else:
        img_out = (x * 255.0 + 0.5).astype(np.uint8)
        img_out = np.ascontiguousarray(img_out)
        Image.fromarray(img_out).save(out_path)


def get_next_run_dir(root_out_dir: str) -> str:
    """
    在 root_out_dir 下自动创建 0/1/2/... 这样的运行编号文件夹
    返回本次运行文件夹路径
    """
    os.makedirs(root_out_dir, exist_ok=True)

    existing = []
    for name in os.listdir(root_out_dir):
        p = os.path.join(root_out_dir, name)
        if os.path.isdir(p) and name.isdigit():
            existing.append(int(name))

    next_id = 0 if len(existing) == 0 else (max(existing) + 1)
    run_dir = os.path.join(root_out_dir, str(next_id))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="输入PNG文件夹")
    parser.add_argument("--out_dir", type=str, required=True, help="输出根目录（脚本会在里面自动建0/1/2...）")
    parser.add_argument("--ext", type=str, default=".png", help="图片后缀（默认 .png）")
    parser.add_argument("--win", type=int, default=5, help="空间窗口大小（默认5，对应5x5）")
    parser.add_argument("--twin", type=int, default=1, help="时间窗口帧数（此版本=1，只用单帧）")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon 防止除零")
    parser.add_argument("--bitdepth", type=int, default=16, choices=[8, 16], help="输出PNG位深（默认16）")
    parser.add_argument("--save_npy", type=int, default=1, help="是否同时保存float32的npy（1=是,0=否）")
    parser.add_argument("--preview_first", type=int, default=1, help="是否弹窗预览第一张（1=是,0=否）")
    args = parser.parse_args()

    if args.twin != 1:
        raise ValueError("你说要“使用一帧计算衬比”，所以此版本要求 --twin 必须是 1。")

    # ====== 改动：本次运行输出目录 = out_dir/0(or1or2...) ======
    run_dir = get_next_run_dir(args.out_dir)
    png_dir = os.path.join(run_dir, "png")
    npy_dir = os.path.join(run_dir, "npy")
    os.makedirs(png_dir, exist_ok=True)
    if args.save_npy == 1:
        os.makedirs(npy_dir, exist_ok=True)

    files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(args.ext)]
    if len(files) == 0:
        raise RuntimeError(f"在 {args.in_dir} 没找到后缀为 {args.ext} 的图片。")

    files.sort(key=natural_key)
    paths = [os.path.join(args.in_dir, f) for f in files]
    n = len(paths)
    print(f"[INFO] 找到 {n} 张图片，将计算每一帧的 1/K^2（仅单帧5x5空间窗口）")

    formula = (
        "公式（单帧5x5空间窗口）：\n"
        "μ = mean_5x5(I)\n"
        "m2 = mean_5x5(I^2)\n"
        "Var = m2 - μ^2\n"
        "K^2 = Var / (μ^2 + ε)\n"
        "显示：1/K^2 ≈ μ^2 / (Var + ε)\n"
        f"win={args.win}, twin=1, ε={args.eps:g}\n"
        "说明：不做任何百分位裁剪（不使用percentile），PNG保存采用 min/max 线性拉伸。\n"
    )
    print(formula)

    print(f"[INFO] 本次运行目录: {run_dir}")
    print(f"[INFO] PNG 输出目录: {png_dir}")
    if args.save_npy == 1:
        print(f"[INFO] NPY 输出目录: {npy_dir}")

    # 记录一下本次配置（方便回溯）
    with open(os.path.join(run_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"in_dir: {args.in_dir}\n")
        f.write(f"ext: {args.ext}\n")
        f.write(f"win: {args.win}\n")
        f.write(f"twin: {args.twin}\n")
        f.write(f"eps: {args.eps}\n")
        f.write(f"bitdepth: {args.bitdepth}\n")
        f.write(f"save_npy: {args.save_npy}\n")
        f.write(formula + "\n")

    box_mean_2d = get_box_mean_filter_2d()

    cache = OrderedDict()
    cache_max = 4

    def get_frame(idx: int) -> np.ndarray:
        idx = max(0, min(n - 1, idx))
        if idx in cache:
            cache.move_to_end(idx)
            return cache[idx]
        img = read_png_as_float32(paths[idx])
        cache[idx] = img
        cache.move_to_end(idx)
        if len(cache) > cache_max:
            cache.popitem(last=False)
        return img

    preview_done = False

    for t in range(n):
        img = get_frame(t)
        inv_k2 = compute_invK2_single_frame(img, box_mean_2d, args.win, args.eps)

        base = os.path.splitext(files[t])[0]

        out_png = os.path.join(png_dir, f"{base}_invK2.png")
        save_vis_png_no_percentile(inv_k2, out_png, args.bitdepth)

        if args.save_npy == 1:
            out_npy = os.path.join(npy_dir, f"{base}_invK2.npy")
            np.save(out_npy, inv_k2.astype(np.float32))

        if (args.preview_first == 1) and (not preview_done):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            plt.imshow(inv_k2, cmap="gray", vmin=inv_k2.min(), vmax=inv_k2.max())
            plt.title("Display: 1/K^2 (single-frame 5x5)")
            plt.axis("off")
            plt.gcf().text(
                0.02, 0.02, formula,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
            )
            plt.show()
            preview_done = True

        if (t + 1) % 50 == 0 or (t + 1) == n:
            print(f"[INFO] 已处理 {t+1}/{n}")

    print(f"[DONE] 本次输出目录：{run_dir}")
    print(f"       ├─ png/: {png_dir}")
    if args.save_npy == 1:
        print(f"       └─ npy/: {npy_dir}")


if __name__ == "__main__":
    main()
