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

# 准备样本和标签批次
sample_batches = []  # (B,τ,4)
for i in range(0,len(samples),B):
    sample_batches.append(samples[i:i+B])
label_batches = []  # (B,)
for i in range(0,len(labels),B):
    label_batches.append(labels[i:i+B])

# 超参数：嵌入维度 d_model
d_model = 128

# 超参数：输入维度 d_in
d_in = 4

# Xavier 初始化权重和偏置
W_e = np.random.randn(d_in,d_model)*np.sqrt(1.0/d_in)  # (4,d_model)
b_e = np.zeros(d_model)                                # (d_model,)

# 线性投影
E = []  # (B,τ,d_model)
for _,X in enumerate(sample_batches):
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

# Xavier 初始化回归头权重和偏置
W_pred = np.random.randn(d_model,1)*np.sqrt(1.0/d_model)
b_pred = np.zeros(1)

# 存储预测值 (B,)
predictions = []
# 存储真实标签 (B,)
true_labels = []

# 回归预测头
for i,out in enumerate(outs_ENC):
    out = np.array(out)  # (B,τ,d_model)
    label = np.array(label_batches[i])  # (B,)
    # 取最后一步
    repr = out[:,-1,:]  # (B,d_model)
    # 应用回归头
    y_pred = np.matmul(repr,W_pred)+b_pred  # (B_i, 1)
    y_pred = y_pred.squeeze(-1)  # (B_i,)
    # 储存
    predictions.append(y_pred)
    true_labels.append(label)
    
# 计算平均 MSE 损失
total_loss = 0.0
total_samples = 0
for y_pred,y_true in zip(predictions,true_labels):
    total_loss += np.mean((y_pred-y_true)**2)*len(y_true)
    total_samples += len(y_true)
mse_loss = total_loss/total_samples

print(f"Initial MSE Loss: {mse_loss:.6f}")

assert isinstance(mse_loss, float), "Loss should be a scalar"
print("✅ Loss computed successfully.")
