import numpy as np
import pandas as pd

# 读取训练数据集
raw_data = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 序列窗口大小超参数 τ
# todo: 根据系统辨识的 CRMS 估计取优？
tau = 10

# 切分滑动窗口得到训练样本
tmp_samples = []
for i in range(raw_data.shape[0]-tau+1):
    sample = raw_data[i:i+tau,:]
    tmp_samples.append(sample)
samples = np.array(tmp_samples)
del tmp_samples

print(samples.shape)