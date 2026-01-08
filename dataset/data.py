import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from glob import glob

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from glob import glob


class SpeckleN2NLogDataset(Dataset):
    def __init__(self, root_dir, crop_size=512, intervals=[5, 7, 9]):
        """
        Args:
            root_dir (str): 大文件夹路径。
            crop_size (int): 中心裁剪尺寸。
            intervals (list): 随机帧间隔。
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.intervals = intervals
        self.max_interval = max(intervals)
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        subfolders = [f.path for f in os.scandir(self.root_dir) if f.is_dir()]

        # 如果根目录下没有子文件夹，可能根目录本身就是包含.npy的文件夹
        # 增加一个兼容逻辑（可选），但根据你的描述是有很多子文件夹的
        if not subfolders:
            # 尝试直接在 root_dir 找
            subfolders = [self.root_dir]

        for folder in subfolders:
            files = sorted(glob(os.path.join(folder, '*.npy')))
            num_frames = len(files)

            if num_frames <= self.max_interval:
                continue

            for i in range(num_frames - self.max_interval):
                self.samples.append((files, i))

        print(f"数据准备完成。共找到 {len(self.samples)} 个样本对 (Log域训练)。")

    def _center_crop(self, img):
        h, w = img.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            # 如果图片小于裁剪尺寸，不报错，而是直接返回原图（或者你需要抛出异常）
            # 这里为了稳健性，如果小于尺寸就不裁了，或者你可以选择 raise ValueError
            return img

        start_h = (h - self.crop_size) // 2
        start_w = (w - self.crop_size) // 2
        return img[start_h: start_h + self.crop_size, start_w: start_w + self.crop_size]

    # ========================================================
    #  修复点：添加 __len__ 方法
    # ========================================================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_list, frame_idx = self.samples[idx]
        delta = random.choice(self.intervals)

        # 读取数据
        input_img = np.load(file_list[frame_idx]).astype(np.float32)
        target_img = np.load(file_list[frame_idx + delta]).astype(np.float32)

        # 中心裁剪
        input_img = self._center_crop(input_img)
        target_img = self._center_crop(target_img)

        # Log1p 归一化
        input_img = np.log1p(input_img)
        target_img = np.log1p(target_img)

        # 转 Tensor [C, H, W]
        input_tensor = torch.from_numpy(input_img).unsqueeze(0)
        target_tensor = torch.from_numpy(target_img).unsqueeze(0)

        return input_tensor, target_tensor


# -------------------------------------------------------------------
# 验证代码
# -------------------------------------------------------------------
if __name__ == '__main__':
    # 模拟数据
    # 假设你的数据是正值，如果原始数据有负值，log1p会报错或产生NaN，请确保数据>=0
    dataset = SpeckleN2NLogDataset(root_dir=r"你的数据路径", crop_size=512)

    # 获取一个样本看看数值范围
    if len(dataset) > 0:
        inp, tar = dataset[0]
        print(f"Log域数据范围: Min={inp.min():.4f}, Max={inp.max():.4f}")

        # 验证还原
        # 训练完推理时，你需要用 expm1 还原： x = exp(y) - 1
        restored = torch.expm1(inp)
        print(f"还原后数据范围: Min={restored.min():.4f}, Max={restored.max():.4f}")