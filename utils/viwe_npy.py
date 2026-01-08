import numpy as np
import matplotlib.pyplot as plt

# 1. 加载 .npy 文件
file_path = "/home/songyd/Projects/lightweight_G/results_raw/data_npy/0_denoised.npy"  # 替换成你的文件路径
data = np.load(file_path)

# 2. 打印形状和数据类型 (这一步很重要，防止维度不对报错)
print(f"数据形状 (Shape): {data.shape}")
print(f"数据类型 (Dtype): {data.dtype}")
print(f"最大值: {data.max()}, 最小值: {data.min()}")

# 3. 如果数据维度不对，进行调整 (Squeeze)
# 比如你的模型输出可能是 (1, 1, 512, 512) 或 (1, 512, 512)，matplotlib 只接受 (H, W)
if data.ndim > 2:
    data = np.squeeze(data)  # 自动把所有维数为1的维度去掉
    print(f"调整后形状: {data.shape}")

# 4. 使用 Matplotlib 显示
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='gray')  # cmap='gray' 表示以灰度图显示
plt.colorbar(label='Pixel Value')  # 显示色条，看数值范围
plt.title(f"View: {file_path}")
plt.axis('off')  # 不显示坐标轴
plt.show()