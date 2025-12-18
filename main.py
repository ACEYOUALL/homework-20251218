import numpy as np
import pandas as pd

# （1）数据预处理

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
B = 16
batches = []
for i in range(0, len(samples), B):
    batch = samples[i:i+B]
    batches.append(batch)

# （2）特征嵌入层

# 超参数：嵌入维度
d_model = 64
# 原始输入维度
d_raw = 5
'''d_raw = raw_data.shape[1]'''

# Xavier 方法初始化权重，标准差 sqrt(1/5)
W_ref = np.random.randn(d_raw, d_model)*np.sqrt(1.0/d_raw)
# 初始化偏置，全零
b_ref = np.zeros(d_model)

# 内嵌线性投影矩阵
embedded_E = []
for _,X in enumerate(batches):
    # 做线性投影
    E = X@W_ref+b_ref
    embedded_E.append(E)

# Transformer 标准的位置编码方式

# 生成位置索引，增加维数到 τ×1
pos = np.arange(tau)[:, np.newaxis]
# 序，长度 d_model/2
i = np.arange(0, d_model, 2)
# 分母
div_term = np.exp(i*(-np.log(10000.0)/d_model))
# 位置编码矩阵，维数 τ×d_model
PE = np.zeros((tau, d_model))
PE[:, 0::2] = np.sin(pos*div_term)  # 偶数维度
PE[:, 1::2] = np.cos(pos*div_term)  # 奇数维度

print(PE.shape)
print(PE[0, :5])
print(PE[1, :5])
