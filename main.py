import numpy as np
import pandas as pd

# 读取训练序列
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 分离标签
seq_X = seq[:,:4]
seq_Y = seq[:,4]

# Z-score 归一化
mean_X, std_X = seq_X.mean(axis=0), seq_X.std(axis=0)
mean_Y, std_Y = seq_Y.mean(), seq_Y.std()
norm_seq_X = (seq_X-mean_X)/(std_X+1e-8)
norm_seq_Y = (seq_Y-mean_Y)/(std_Y+1e-8)

# 超参数：滑动窗口长度 τ
tau = 10

# 样本和标签
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(norm_seq_X)-tau):
    samples.append(norm_seq_X[i:i+tau,:])
    labels.append(norm_seq_Y[i+tau])

# 超参数：批量 B
B = 32

# 分批
batches = []  # (B,τ,4)
for i in range(0,len(samples),B):
    batches.append(samples[i:i+B])

# 超参数：嵌入维度 d_model
d_model = 128

# 超参数：输入维度 d_in
d_in = 4

# Xavier 初始化权重和偏置
W_e = np.random.randn(d_in,d_model)*np.sqrt(1.0/d_in)  # (4,d_model)
b_e = np.zeros(d_model)                                # (d_model,)

# 线性投影
E = []  # (B,τ,d_model)
for _,X in enumerate(batches):
    E.append(X@W_e+b_e)

# 位置编码
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))  # (τ,d_model)
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

# 注入编码
Z = []  # (B,τ,d_model)
for _,E in enumerate(E):
    Z.append(E+P)

# 层归一化
def LayerNorm(Z, gamma, beta):
    mean = np.mean(Z,axis=-1,keepdims=True)
    std = np.std(Z,axis=-1,keepdims=True)
    return gamma*((Z-mean)/(std+1e-8))+beta

# 超参数：注意力头数 h
h = 8

# 单头维度
d_K = d_model//h
d_V = d_K

# Xavier 初始化权重，(d_model,d_model)
W_Q = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_K = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_V = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_O = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)

# 缩放点积注意力
def ScaledDotProductAttention(Q_i, K_i, V_i):
    # 注意这里 Q_i、K_i、V_i 是单头，(B,τ,d_K)
    # 注意力得分：QK
    AS = np.matmul(Q_i,K_i.transpose(0,2,1))/np.sqrt(d_K)
    # softmax（数值稳定形式）计算注意力权重，(B,τ,τ)
    AS = AS - np.max(AS,axis=-1,keepdims=True)
    exp_AS = np.exp(AS)
    AW = exp_AS/np.sum(exp_AS,axis=-1,keepdims=True)
    # 单头输出，(B,τ,d_V)
    out = np.matmul(AW,V_i)
    return out,AW

# 多头注意力机制 MHA
def MHA(Z, W_Q, W_K, W_V, W_O):
    B,tau,_ = Z.shape
    # 计算 Q、K、V，(B,τ,d_model)
    Q = np.matmul(Z,W_Q)
    K = np.matmul(Z,W_K)
    V = np.matmul(Z,W_V)
    # 分离 Q、K、V
    Q_iso = Q.reshape(B,tau,h,d_K).transpose(0,2,1,3)
    K_iso = K.reshape(B,tau,h,d_K).transpose(0,2,1,3)
    V_iso = V.reshape(B,tau,h,d_V).transpose(0,2,1,3)
    # 注意力结果，(B,τ,d_V)
    outs = []
    # 注意力权重，(B,τ,τ)
    AWs = []
    # 计算单头注意力
    for i in range(h):
        Q_i = Q_iso[:,i,:,:]
        K_i = K_iso[:,i,:,:]
        V_i = V_iso[:,i,:,:]
        out, AW = ScaledDotProductAttention(Q_i,K_i,V_i)
        outs.append(out)
        AWs.append(AW)
    # 拼接并得到多头结果，(B,τ,d_V·h) → (B,τ,d_model)
    outs_MHA = np.matmul(np.concatenate(outs,axis=-1),W_O)
    return outs_MHA,AWs

# 超参数：中间层维度
d_ff = 8 * d_model

# Xavier 初始化权重
W_1 = np.random.randn(d_model,d_ff)*np.sqrt(1.0/d_model)
b_1 = np.zeros(d_ff)
W_2 = np.random.randn(d_ff,d_model)*np.sqrt(1.0/d_ff)
b_2 = np.zeros(d_model)

# Swish 激活函数
def Swish(x, beta=1.0):
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x*sigmoid

# ReLU 激活函数
def ReLU(x):
    return np.maximum(0,x)

# 前馈网络 FFN
def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1)+b_1
    A = Swish(L_1)
    L_2 = np.matmul(A, W_2)+b_2
    return L_2

# 层归一化参数
gamma = np.ones(d_model)
beta = np.zeros(d_model)

# 编码器层输出
outs_ENC = []

# 两次残差连接
for batch in Z:
    outs_MHA, AWs = MHA(batch,W_Q,W_K,W_V,W_O)
    res_1 = batch+outs_MHA
    outs_LN_1 = LayerNorm(res_1,gamma,beta)
    outs_FFN = FFN(outs_LN_1,W_1,b_1,W_2,b_2)
    res_2 = outs_LN_1+outs_FFN
    outs_LN_2 = LayerNorm(res_2,gamma,beta)
    outs_ENC.append(outs_LN_2)
    
# 转为 array（推荐）
samples = np.array(samples)
labels = np.array(labels)

# 验证 d_model 可被 h 整除
assert d_model % h == 0, "d_model must be divisible by h"

# 测试所有批次（包括最后一个不完整批次）
for i, batch_X in enumerate(batches):
    batch_X = np.array(batch_X)
    B_i = batch_X.shape[0]
    
    # 前向传播
    E_i = batch_X @ W_e + b_e
    Z_i = E_i + P
    out_enc = outs_ENC[i]  # 你的 outs_ENC[i] 应等于下面结果
    
    # 重新计算以验证
    out_mha, _ = MHA(Z_i, W_Q, W_K, W_V, W_O)
    out_l1 = LayerNorm(Z_i + out_mha, gamma, beta)
    out_ffn = FFN(out_l1, W_1, b_1, W_2, b_2)
    out_final = LayerNorm(out_l1 + out_ffn, gamma, beta)
    
    # 形状检查
    assert out_final.shape == (B_i, tau, d_model)
    
    # 验证与 outs_ENC 一致（可选）
    assert np.allclose(out_final, outs_ENC[i], atol=1e-6)

# 验证注意力 softmax
_, aw_list = MHA(Z[0], W_Q, W_K, W_V, W_O)
assert np.allclose(aw_list[0][0].sum(axis=-1), 1.0, atol=1e-5)

print("✅ 编码器前向传播验证通过！")
