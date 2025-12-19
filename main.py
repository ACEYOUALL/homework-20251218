import numpy as np
import pandas as pd

# （1）数据预处理

# 读取训练数据集
raw_data = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# Z-score 归一化（时间维度）
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

# 内嵌线性投影序列，维度 B×τ×d_model
embedded_E = []
for _,X in enumerate(batches):
    # 做线性投影
    E = X@W_ref+b_ref
    embedded_E.append(E)

# Transformer 标准的位置编码方式

# 生成位置索引，增加维数到 τ×1
pos = np.arange(tau)[:, np.newaxis]
# 索引子序，长度 d_model/2
i = np.arange(0, d_model, 2)
# 分母
div_term = np.exp(i*(-np.log(10000.0)/d_model))
# 位置编码序列，维数 τ×d_model
PE = np.zeros((tau, d_model))
PE[:, 0::2] = np.sin(pos*div_term)  # 偶数维度
PE[:, 1::2] = np.cos(pos*div_term)  # 奇数维度

# 注入位置编码

# 输入嵌入序列，维度 B×τ×d_model
embedded_Z = []
for _,E in enumerate(embedded_E):
    # 线性叠加
    Z = E+PE
    embedded_Z.append(Z)

# 回收内存
del raw_data
del norm_data
del batches
del embedded_E

# （3）编码层

# 层归一化（单样本，按特征维度 d_model ）
def layer_norm(Z, gamma, beta):
    mean = np.mean(Z, axis=-1, keepdims=True)
    std = np.std(Z, axis=-1, keepdims=True)
    norm_Z = (Z-mean)/(std+1e-6)
    output = gamma*norm_Z+beta
    return output

# 多头注意力机制

# 超参数：头数（d_model%h==0）
h = 8
# 单头维度
d_k = d_model//h
d_v = d_k

# Xavier 方法初始化权重参数，维度是 d_model×d_model
W_Q = np.random.randn(d_model, d_model)*np.sqrt(1.0/d_model)
W_K = np.random.randn(d_model, d_model)*np.sqrt(1.0/d_model)
W_V = np.random.randn(d_model, d_model)*np.sqrt(1.0/d_model)

# 初始化多头拼接后的输出权重，维度 d_model×d_model
W_O = np.random.randn(d_model, d_model)*np.sqrt(1.0/d_model)

# 缩放点积注意力（查询、键、值）
def scaled_dot_product_attention(Q, K, V):
    # 注意力得分 QK^T，K 转置最后两维
    attention_scores = np.matmul(Q, K.transpose(0,2,1))
    # 缩放，避免 attention_score 太大导致 softmax 梯度消失
    attention_scores = attention_scores/np.sqrt(d_k)
    # softmax 计算注意力权重，并归一化（按最后一维，权重之和等于 1 ），维度是 B×τ×τ
    attention_weights = np.exp(attention_scores)/np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    # 权重乘以 V（维度是 B×τ×d_v ），得到单头输出，维度是 B×τ×d_v
    output = np.matmul(attention_weights, V)
    return output,attention_weights

# 多头注意力实现部分
def multi_head_attention(Z, W_Q, W_K, W_V, W_O):
    #  W_Q、W_K、W_V、W_O 维度均是 d_model×d_model
    
    # 获取批次大小和时间步
    B,tau,_ = Z.shape
    
    # 计算 Q、K、V ，维度是 B×τ×d_model
    Q = np.matmul(Z,W_Q)
    K = np.matmul(Z,W_K)
    V = np.matmul(Z,W_V)
    
    # 将每个头分离开来，变成维度是 B×h×τ×d_k
    Q_split = Q.reshape(B,tau,h,d_k).transpose(0,2,1,3)
    K_split = K.reshape(B,tau,h,d_k).transpose(0,2,1,3)
    V_split = V.reshape(B,tau,h,d_v).transpose(0,2,1,3)
    
    # 单头注意力结果
    head_outputs = []
    # 每个头的注意力权重
    attention_weights = []
    
    # 遍历每个头
    for i in range(h):
        
        # 维度是 B×τ×d_k
        Q_i = Q_split[:,i,:,:]
        K_i = K_split[:,i,:,:]
        V_i = V_split[:,i,:,:]
        
        # 计算单头注意力
        head_outputs_, attention_weights_ = scaled_dot_product_attention(Q_i, K_i, V_i)
        head_outputs.append(head_outputs_)            # B×τ×d_v
        attention_weights.append(attention_weights_)  # B×τ×τ
        
    # 将多头结果拼接
    concat_head_outputs = np.concatenate(head_outputs, axis=-1)  # B×τ×(d_v·h) → B×τ×d_model
    
    # 最后乘 W_O 线性变换整合
    multi_head_outputs = np.matmul(concat_head_outputs, W_O)  # B×τ×d_model
    
    return multi_head_outputs,attention_weights

# 前馈网络

# 超参数：中间层维度（取 8 倍 d_model，满足 d_ff >> d_model）
d_ff = 512

# Xavier 方法初始化第一层权重
W1 = np.random.randn(d_model, d_ff)*np.sqrt(1.0/d_model)
# 初始化第一层偏置，全零
b1 = np.zeros(d_ff)

# Xavier 方法初始化第二层权重
W2 = np.random.randn(d_ff, d_model)*np.sqrt(1.0 / d_ff)
# 初始化第二层偏置，全零
b2 = np.zeros(d_model)

# Swish 激活函数
def swish(x, beta=1.0):
    sigmoid = 1/(1+np.exp(-beta*x))
    return x*sigmoid

# ReLU 激活函数
def relu(x):
    return np.maximum(0, x)

# 实现前馈网络
def feed_forward_network(Z, W1, b1, W2, b2):
    # 线性变换，维度拓展至 d_ff
    L1 = np.matmul(Z, W1)+b1
    # 激活函数
    A = swish(L1)
    # 线性变换，维度压缩回 d_model
    L2 = np.matmul(A, W2)+b2
    return L2

# 残差连接

# 层归一化参数
ln_gamma = np.ones(d_model)
ln_beta = np.zeros(d_model)

# 编码层输出
encoder_outputs = []

print()

# Transformer 编码层含两次残差连接
for batch in embedded_Z:
    # 第一次
    # 计算多头注意力（子层输出）
    multi_head_outputs,attention_weights = multi_head_attention(batch, W_Q, W_K, W_V, W_O)
    # 残差连接，B×τ×d_model
    residual1 = batch+multi_head_outputs
    # 层归一化
    ln1_output = layer_norm(residual1, ln_gamma, ln_beta)
    
    # 第二次
    # 计算前馈网络（子层输出）
    ffn_outputs = feed_forward_network(ln1_output, W1, b1, W2, b2)
    # 残差连接，B×τ×d_model
    residual2 = ln1_output+ffn_outputs
    # 层归一化
    ln2_output = layer_norm(residual2, ln_gamma, ln_beta)
    
    encoder_outputs.append(ln2_output)
