import numpy as np
import pandas as pd

# 读取训练数据集
raw_data = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# Z-score 标准化
mean = np.mean(raw_data, axis=0)
std = np.std(raw_data, axis=0)
norm_data = (raw_data-mean)/(std+1e-8)

# 超参数：序列窗口大小 τ
# todo: 根据系统辨识的 CRMS 估计取优？
tau = 10

# 切分滑动窗口得到训练样本
tmp_samples = []
for i in range(norm_data.shape[0]-tau+1):
    sample = norm_data[i:i+tau,:]
    tmp_samples.append(sample)
samples = np.array(tmp_samples)
del tmp_samples

# 超参数：分批次 B（考虑到物理内存 12GB 的限制）
B = 32
batches = []
for i in range(0, len(samples), B):
    batch = samples[i:i+B]
    batches.append(batch)
