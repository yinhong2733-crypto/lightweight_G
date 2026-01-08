# speckle_invK2_folder.py
# 输入：一个文件夹（999张png）
# 输出：每帧一张 1/K^2 图（背景黑、血管白）
# 计算：5x5 空间窗口 + 5 帧时间窗口（5x5x5）

import argparse
import os
import re
from collections import OrderedDict

import numpy as np
from PIL import Image


# -----------------------------
# 1) 文件名自然排序（1.png, 2.png, 10.png 不会乱）
# -----------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# -----------------------------
# 2) 读 PNG（尽量保留原始 bit depth）
#    - 灰度图：8bit/16bit 都支持
#    - RGB：转灰度（会变8bit，若你原本是RGB那只能这样）
# -----------------------------
def read_png_as_float32(path: str) -> np.ndarray:
    img = Image.open(path)

    # 如果是彩色，转灰度（注意：会变成8bit）
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")

    arr = np.array(img)

    # 统一转 float32 计算
    arr = arr.astype(np.float32)
    return arr


# -----------------------------
# 3) 2D 5x5 box mean filter（均值滤波）
#    优先用 scipy（快），没有就用 torch（也能跑）
# -----------------------------
def get_box_mean_filter_2d():
    # 尝试 scipy
    try:
        from scipy.ndimage import uniform_filter

        def box_mean(img2d: np.ndarray, win: int) -> np.ndarray:
            # uniform_filter 本身就是均值滤波
            return uniform_filter(img2d, size=win, mode="reflect")

        return box_mean
    except Exception:
        pass

    # 尝试 torch
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


# -----------------------------
# 4) 计算单帧的 1/K^2（使用 t-2..t+2 五帧）
#    关键：5x5x5 的均值等价于：
#    先每帧做 5x5 空间均值，再在 5 帧上取平均（同理对 I^2）
# -----------------------------
def compute_invK2_for_t(frames5: list[np.ndarray], box_mean_2d, win: int, eps: float) -> np.ndarray:
    # 对每帧计算：空间均值 mean(I) 和 mean(I^2)
    means = []
    means2 = []
    for img in frames5:
        m = box_mean_2d(img, win)
        m2 = box_mean_2d(img * img, win)
        means.append(m)
        means2.append(m2)

    # 再在 5 帧上取平均 => 就是 5x5x5 的 μ 和 m2
    mu = np.mean(means, axis=0)
    m2 = np.mean(means2, axis=0)

    var = m2 - mu * mu
    var = np.maximum(var, 0.0)  # 防止数值误差出现负数

    # 你要的：1/K^2  (更稳写法：mu^2/(var+eps))
    inv_k2 = (mu * mu) / (var + eps)
    return inv_k2


# -----------------------------
# 5) 保存：把 invK2 映射到 8bit 或 16bit PNG
#    使用百分位裁剪，让“血管更白、背景更黑”更明显
# -----------------------------
def save_vis_png(inv_k2: np.ndarray, out_path: str, p_lo: float, p_hi: float, bitdepth: int):
    lo = np.percentile(inv_k2, p_lo)
    hi = np.percentile(inv_k2, p_hi)
    if hi <= lo:
        hi = lo + 1e-6

    x = np.clip(inv_k2, lo, hi)
    x = (x - lo) / (hi - lo)

    if bitdepth == 16:
        img_out = (x * 65535.0 + 0.5).astype(np.uint16)
        Image.fromarray(img_out, mode="I;16").save(out_path)
    else:
        img_out = (x * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(img_out, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="输入PNG文件夹（里面999张png）")
    parser.add_argument("--out_dir", type=str, required=True, help="输出文件夹（保存1/K^2结果）")
    parser.add_argument("--ext", type=str, default=".png", help="图片后缀（默认 .png）")
    parser.add_argument("--win", type=int, default=5, help="空间窗口大小（默认5，对应5x5）")
    parser.add_argument("--twin", type=int, default=5, help="时间窗口帧数（必须5）")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon 防止除零")
    parser.add_argument("--p_lo", type=float, default=1.0, help="显示裁剪低百分位（默认1）")
    parser.add_argument("--p_hi", type=float, default=99.0, help="显示裁剪高百分位（默认99）")
    parser.add_argument("--bitdepth", type=int, default=16, choices=[8, 16], help="输出PNG位深（默认16）")
    parser.add_argument("--save_npy", type=int, default=0, help="是否同时保存float32的npy（1=是,0=否）")
    parser.add_argument("--preview_first", type=int, default=1, help="是否弹窗预览第一张（1=是,0=否）")
    args = parser.parse_args()

    if args.twin != 5:
        raise ValueError("本脚本按你的要求固定使用 5 帧时间窗口（--twin 必须是 5）。")

    os.makedirs(args.out_dir, exist_ok=True)

    # 列出所有png
    files = [f for f in os.listdir(args.in_dir) if f.lower().endswith(args.ext)]
    if len(files) == 0:
        raise RuntimeError(f"在 {args.in_dir} 没找到后缀为 {args.ext} 的图片。")

    files.sort(key=natural_key)
    paths = [os.path.join(args.in_dir, f) for f in files]
    n = len(paths)
    print(f"[INFO] 找到 {n} 张图片，将计算每一帧的 1/K^2（5x5空间 + 5帧时间）")

    # 公式打印
    formula = (
        "公式：\n"
        "μ = mean(I)  (在 5x5x5 内)\n"
        "m2 = mean(I^2)\n"
        "Var = m2 - μ^2\n"
        "K^2 = Var / (μ^2 + ε)\n"
        "显示：1/(K^2+ε) ≈ μ^2 / (Var + ε)\n"
        f"win={args.win}, twin=5, ε={args.eps:g}\n"
    )
    print(formula)

    # 均值滤波函数
    box_mean_2d = get_box_mean_filter_2d()

    # 做一个小缓存：避免反复读同一张（滑窗会复用很多帧）
    cache = OrderedDict()
    cache_max = 12  # 足够覆盖滑动窗口

    def get_frame(idx: int) -> np.ndarray:
        # 边界复制（replicate）：idx <0 用0，idx>=n 用n-1
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

    # 主循环：对每一帧 t 计算 1/K^2
    preview_done = False

    for t in range(n):
        idxs = [t - 2, t - 1, t, t + 1, t + 2]
        frames5 = [get_frame(i) for i in idxs]

        inv_k2 = compute_invK2_for_t(frames5, box_mean_2d, args.win, args.eps)

        base = os.path.splitext(files[t])[0]
        out_png = os.path.join(args.out_dir, f"{base}_invK2.png")
        save_vis_png(inv_k2, out_png, args.p_lo, args.p_hi, args.bitdepth)

        if args.save_npy == 1:
            out_npy = os.path.join(args.out_dir, f"{base}_invK2.npy")
            np.save(out_npy, inv_k2.astype(np.float32))

        if (args.preview_first == 1) and (not preview_done):
            # 只预览第一张
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            # 用同样的裁剪规则显示
            lo = np.percentile(inv_k2, args.p_lo)
            hi = np.percentile(inv_k2, args.p_hi)
            plt.imshow(np.clip(inv_k2, lo, hi), cmap="gray", vmin=lo, vmax=hi)
            plt.title("Display: 1/K^2  (background dark, vessels bright)")
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

    print(f"[DONE] 输出已保存到：{args.out_dir}")


if __name__ == "__main__":
    main()
