import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# (1) 数据预处理
# ------------------------------------------------------------------

# 读取原始数据，仅保留需要的特征列 (AT, EV, AP, RH) 和目标列 (PE)
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 分离特征和标签（原始未归一化数据）
# seq_X: [样本数, 4] - 4个特征
# seq_Y: [样本数]   - 目标值(PE)
seq_X = seq[:,:4]
seq_Y = seq[:,4]

# 超参数：滑动窗口长度 τ (时间步数)
tau = 16

# 构建时序样本：每个样本包含τ个连续时间步的特征，预测下一个时间步的PE值
# samples: [样本数, τ, 4] - 时序特征
# labels: [样本数]        - 对应样本τ+1时刻的PE值
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(seq_X)-tau):
    samples.append(seq_X[i:i+tau,:])
    labels.append(seq_Y[i+tau])

# 时序数据划分：前80%为训练集，后20%为验证集（避免随机打乱破坏时序依赖性）
split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
train_labels = labels[:split_idx]
val_samples = samples[split_idx:]
val_labels = labels[split_idx:]

# 仅基于训练集计算归一化统计量（防止数据泄露）
train_samples_np = np.array(train_samples)  # [N_train, τ, 4]
train_labels_np = np.array(train_labels)    # [N_train]

# 计算训练集特征的均值/标准差（展平为2D计算，保持特征维度一致性）
# mean_X_train: [4] - 每个特征通道的均值
# std_X_train: [4]  - 每个特征通道的标准差
mean_X_train = train_samples_np.reshape(-1, 4).mean(axis=0)
std_X_train = train_samples_np.reshape(-1, 4).std(axis=0)
# 计算训练集标签的均值/标准差
mean_Y_train = train_labels_np.mean()
std_Y_train = train_labels_np.std()

# 用训练集统计量归一化训练集和验证集
norm_train_samples = (train_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_train_labels = (train_labels_np - mean_Y_train) / (std_Y_train + 1e-8)
val_samples_np = np.array(val_samples)
val_labels_np = np.array(val_labels)
norm_val_samples = (val_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_val_labels = (val_labels_np - mean_Y_train) / (std_Y_train + 1e-8)

print(f"数据集划分完成 - 训练样本数: {len(norm_train_samples)}, 验证样本数: {len(norm_val_samples)}")

# 超参数：批量大小 B
B = 32

# 生成训练集批次
train_sample_batches = []
for i in range(0, len(norm_train_samples), B):
    train_sample_batches.append(norm_train_samples[i:i+B])
train_label_batches = []
for i in range(0, len(norm_train_labels), B):
    train_label_batches.append(norm_train_labels[i:i+B])

# 生成验证集批次
val_sample_batches = []
for i in range(0, len(norm_val_samples), B):
    val_sample_batches.append(norm_val_samples[i:i+B])
val_label_batches = []
for i in range(0, len(norm_val_labels), B):
    val_label_batches.append(norm_val_labels[i:i+B])

# ------------------------------------------------------------------
# (2) 模型参数初始化 - 三层Pre-LN Transformer架构
# 设计要点:
#   - Pre-LN架构：LayerNorm在残差连接之前，提升训练稳定性
#   - 层级学习率：浅层用较高学习率，深层/回归头用较低学习率
#   - Kaiming初始化：适配Swish激活函数，增益因子调整为1/sqrt(fan_in)
#   - 位置编码：固定正弦/余弦编码，不参与训练
# ------------------------------------------------------------------

# 超参数定义
d_model = 64      # 模型维度 (隐藏层大小)
d_in = 4          # 输入特征维度
num_layers = 3    # Transformer编码器层数
h = 16            # 多头注意力头数
d_K = d_model // h  # 单头Key/Query维度
d_V = d_K         # 单头Value维度
d_ff = 4 * d_model  # 前馈网络隐藏层维度

# Kaiming初始化（适配Swish激活函数）
def KaimingInit(shape, fan_in):
    """初始化权重，标准差 = 1/sqrt(fan_in)，适配Swish激活"""
    return np.random.randn(*shape) * np.sqrt(1.0 / fan_in)

# 特征嵌入层：将输入特征投影到模型维度
# W_e: [d_in, d_model] - 特征投影矩阵
# b_e: [d_model]       - 偏置项
W_e = KaimingInit((d_in, d_model), d_in)
b_e = np.zeros(d_model)

# 位置编码 (固定，不训练)
# P: [τ, d_model] - 位置编码矩阵
t = np.arange(tau)[:, np.newaxis]  # [τ, 1]
i = np.arange(0, d_model, 2)       # [d_model/2]
div_term = np.exp(i * (-np.log(10000.0) / d_model))  # 波长缩放因子
P = np.zeros((tau, d_model))
P[:, 0::2] = np.sin(t * div_term)  # 偶数维度用sin
P[:, 1::2] = np.cos(t * div_term)  # 奇数维度用cos

# 参数字典初始化（每层独立参数）
params = {
    'W_e': W_e,  # [4, 64]
    'b_e': b_e,  # [64]
    'W_pred': KaimingInit((d_model, 1), d_model),  # [64, 1] - 回归头权重
    'b_pred': np.array([0.0])                      # [1]    - 回归头偏置
}

# 为每层Transformer初始化独立参数
for layer_idx in range(num_layers):
    # 多头注意力(MHA)参数
    params[f'layer{layer_idx}_W_Q'] = KaimingInit((d_model, d_model), d_model)  # [64,64]
    params[f'layer{layer_idx}_W_K'] = KaimingInit((d_model, d_model), d_model)  # [64,64]
    params[f'layer{layer_idx}_W_V'] = KaimingInit((d_model, d_model), d_model)  # [64,64]
    params[f'layer{layer_idx}_W_O'] = KaimingInit((d_model, d_model), d_model)  # [64,64]
    
    # 前馈网络(FFN)参数
    params[f'layer{layer_idx}_W_1'] = KaimingInit((d_model, d_ff), d_model)   # [64,256]
    params[f'layer{layer_idx}_b_1'] = np.zeros(d_ff)                          # [256]
    params[f'layer{layer_idx}_W_2'] = KaimingInit((d_ff, d_model), d_ff)      # [256,64]
    params[f'layer{layer_idx}_b_2'] = np.zeros(d_model)                       # [64]
    
    # LayerNorm参数 (Pre-LN架构)
    params[f'layer{layer_idx}_gamma1'] = np.ones(d_model)  # [64] - MHA前LayerNorm缩放
    params[f'layer{layer_idx}_beta1'] = np.zeros(d_model)  # [64] - MHA前LayerNorm偏移
    params[f'layer{layer_idx}_gamma2'] = np.ones(d_model)  # [64] - FFN前LayerNorm缩放
    params[f'layer{layer_idx}_beta2'] = np.zeros(d_model)  # [64] - FFN前LayerNorm偏移

# ------------------------------------------------------------------
# (3) 核心组件实现
# 设计要点:
#   - Dropout：训练时随机丢弃，推理时缩放
#   - LayerNorm：归一化特征维度，提升训练稳定性
#   - Scaled Dot-Product Attention：带数值稳定性的注意力计算
#   - Swish激活：平滑非线性激活函数，优于ReLU
# ------------------------------------------------------------------

def Dropout(Z, rate=0.1, training=True):
    """
    Dropout正则化层
    参数:
        Z: [B, τ, d_model] - 输入张量
        rate: 丢弃率
        training: 是否训练模式
    返回:
        Z_dropped: [B, τ, d_model] - Dropout后输出
        mask: [B, τ, d_model] - 丢弃掩码 (训练时)
    """
    if not training or rate <= 0.0:
        return Z, None
    
    # 生成掩码 (1=保留, 0=丢弃)
    mask = (np.random.rand(*Z.shape) > rate).astype(np.float32)
    scale = 1.0 / (1.0 - rate)  # 期望保持缩放
    mask_scaled = mask * scale
    
    Z_dropped = Z * mask_scaled
    return Z_dropped, mask

def LayerNorm(Z, gamma, beta):
    """
    层归一化 (Layer Normalization)
    公式: γ * (Z - μ) / σ + β
    参数:
        Z: [B, τ, d_model] - 输入张量
        gamma: [d_model] - 缩放参数
        beta: [d_model] - 偏移参数
    返回:
        [B, τ, d_model] - 归一化后输出
    """
    mean = np.mean(Z, axis=-1, keepdims=True)  # [B, τ, 1]
    std = np.std(Z, axis=-1, keepdims=True)    # [B, τ, 1]
    return gamma * ((Z - mean) / (std + 1e-8)) + beta

def LayerNorm_with_grad(Z, gamma, beta, dL_dout=None):
    """
    带梯度计算的层归一化 (用于反向传播)
    参数:
        Z: [B, τ, d_model] - 输入
        gamma/beta: [d_model] - 可学习参数
        dL_dout: [B, τ, d_model] - 输出梯度
    返回:
        out: [B, τ, d_model] - 归一化输出
        grads: (dL_dZ, dL_dgamma, dL_dbeta) - 梯度元组
    """
    mean = np.mean(Z, axis=-1, keepdims=True)  # [B, τ, 1]
    std = np.std(Z, axis=-1, keepdims=True)    # [B, τ, 1]
    norm_Z = (Z - mean) / (std + 1e-8)         # [B, τ, d_model]
    out = gamma * norm_Z + beta                # [B, τ, d_model]
    
    if dL_dout is None:
        return out, None
    
    B, T, D = Z.shape
    # 计算gamma/beta梯度
    dL_dgamma = np.sum(dL_dout * norm_Z, axis=(0,1))  # [d_model]
    dL_dbeta = np.sum(dL_dout, axis=(0,1))            # [d_model]
    
    # 计算输入梯度
    dL_dnorm = dL_dout * gamma  # [B, τ, d_model]
    dL_dstd = np.sum(dL_dnorm * (Z - mean) * -0.5 * (std + 1e-8)**-3, axis=-1, keepdims=True)  # [B, τ, 1]
    dL_dmean = np.sum(dL_dnorm * -1.0 / (std + 1e-8), axis=-1, keepdims=True)  # [B, τ, 1]
    dL_dmean += dL_dstd * np.mean(-2.0 * (Z - mean), axis=-1, keepdims=True)   # [B, τ, 1]
    
    dL_dZ = dL_dnorm / (std + 1e-8)  # [B, τ, d_model]
    dL_dZ += dL_dstd * 2.0 * (Z - mean) / D  # [B, τ, d_model]
    dL_dZ += dL_dmean / D  # [B, τ, d_model]
    
    return out, (dL_dZ, dL_dgamma, dL_dbeta)

def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    """
    缩放点积注意力 (单头)
    公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    参数:
        Q_i: [B, τ, d_k] - Query
        K_i: [B, τ, d_k] - Key
        V_i: [B, τ, d_v] - Value (d_v = d_k)
    返回:
        out: [B, τ, d_v] - 注意力输出
        AW: [B, τ, τ] - 注意力权重
        ... (中间变量用于反向传播)
    """
    # 计算注意力分数 [B, τ, τ]
    AS_original = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)
    # 数值稳定化：减去每行最大值
    max_AS = np.max(AS_original, axis=-1, keepdims=True)  # [B, τ, 1]
    AS = AS_original - max_AS
    # 计算注意力权重
    exp_AS = np.exp(AS)  # [B, τ, τ]
    sum_exp_AS = np.sum(exp_AS, axis=-1, keepdims=True)  # [B, τ, 1]
    AW = exp_AS / (sum_exp_AS + 1e-8)  # [B, τ, τ]
    # 加权求和
    out = np.matmul(AW, V_i)  # [B, τ, d_v]
    return out, AW, AS_original, AS, max_AS, sum_exp_AS

def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    """
    多头注意力 (MHA) 实现
    参数:
        Z: [B, τ, d_model] - 输入
        W_Q/K/V/O: [d_model, d_model] - 投影权重
    返回:
        outs_MHA: [B, τ, d_model] - MHA输出
        ... (中间变量用于反向传播)
    """
    B, tau, _ = Z.shape
    # 线性投影
    Q = np.matmul(Z, W_Q)  # [B, τ, d_model]
    K = np.matmul(Z, W_K)  # [B, τ, d_model]
    V = np.matmul(Z, W_V)  # [B, τ, d_model]
    
    # 分头: [B, τ, h, d_k] -> [B, h, τ, d_k]
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    
    # 处理每个注意力头
    outs = []
    AWs = []
    AS_originals = []
    AS_list = []
    max_AS_list = []
    sum_exp_AS_list = []
    V_is = []
    
    for i in range(h):
        Q_i = Q_iso[:,i,:,:]  # [B, τ, d_k]
        K_i = K_iso[:,i,:,:]  # [B, τ, d_k]
        V_i = V_iso[:,i,:,:]  # [B, τ, d_v]
        out, AW, AS_original, AS, max_AS, sum_exp_AS = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)  # [B, τ, d_v]
        AWs.append(AW)
        AS_originals.append(AS_original)
        AS_list.append(AS)
        max_AS_list.append(max_AS)
        sum_exp_AS_list.append(sum_exp_AS)
        V_is.append(V_i)
    
    # 拼接多头输出 [B, τ, h*d_v] = [B, τ, d_model]
    concat_out = np.concatenate(outs, axis=-1)
    # 输出投影 [B, τ, d_model]
    outs_MHA = np.matmul(concat_out, W_O)
    return (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list, 
            V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V)

def Swish(x, beta=1.0):
    """
    Swish激活函数: x * sigmoid(beta*x)
    参数:
        x: 任意形状张量
        beta: 温度参数 (默认1.0)
    """
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x * sigmoid

def FFN(Z, W_1, b_1, W_2, b_2):
    """
    前馈网络 (FFN)
    结构: Linear -> Swish -> Linear
    参数:
        Z: [B, τ, d_model] - 输入
        W_1: [d_model, d_ff] - 第一层权重
        b_1: [d_ff] - 第一层偏置
        W_2: [d_ff, d_model] - 第二层权重
        b_2: [d_model] - 第二层偏置
    返回:
        L_2: [B, τ, d_model] - FFN输出
        L_1: [B, τ, d_ff] - 中间层线性输出
        A: [B, τ, d_ff] - 激活后输出
    """
    L_1 = np.matmul(Z, W_1) + b_1  # [B, τ, d_ff]
    A = Swish(L_1)                 # [B, τ, d_ff]
    L_2 = np.matmul(A, W_2) + b_2  # [B, τ, d_model]
    return L_2, L_1, A

# ------------------------------------------------------------------
# (4) AdamW 优化器
# 设计要点:
#   - 权重衰减与L2正则解耦 (AdamW核心改进)
#   - 分层学习率：回归头用1/10学习率，浅层用1.2x学习率
#   - 梯度裁剪：防止训练不稳定
# ------------------------------------------------------------------
class AdamWOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        AdamW优化器实现
        参数:
            params: 模型参数字典
            lr: 基础学习率
            betas: 一阶/二阶矩估计衰减率
            eps: 数值稳定常数
            weight_decay: 权重衰减系数
        """
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 初始化一阶/二阶矩估计
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}
        self.t = 0
        
        # 识别权重参数 (应用权重衰减)
        self.weight_params = [name for name in params.keys() if 'W_' in name or 'W_e' in name or 'W_pred' in name]
    
    def step(self, grads, lr=None):
        """
        执行单步参数更新
        参数:
            grads: 梯度字典
            lr: 可覆盖基础学习率
        """
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        
        for name in self.params.keys():
            param = self.params[name]
            grad = grads[name]
            
            # 分层学习率策略
            layer_lr = current_lr
            if 'W_pred' in name or 'b_pred' in name:
                layer_lr = current_lr * 0.1  # 回归头用1/10学习率
            elif 'layer0' in name:
                layer_lr = current_lr * 1.2  # 浅层稍高学习率
            
            # 更新矩估计
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # AdamW更新规则
            if name in self.weight_params:
                # 权重衰减独立于梯度
                update = layer_lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)
            else:
                update = layer_lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            
            self.params[name] = param - update
    
    def state_dict(self):
        """保存优化器状态"""
        return {'m': self.m, 'v': self.v, 't': self.t}
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']

# ------------------------------------------------------------------
# (5) 训练循环
# 设计要点:
#   - 余弦退火学习率 + 预热
#   - Pre-LN架构 + Dropout正则化
#   - 梯度裁剪：回归头严格裁剪(-0.01,0.01)，深层中等(-0.05,0.05)，其他(-0.1,0.1)
#   - 早停策略：保存验证损失最低的模型
#   - 梯度健康监控：检查深层/浅层梯度幅值比
# ------------------------------------------------------------------

# 初始化优化器
optimizer = AdamWOptimizer(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4  # 权重衰减系数
)

# 训练超参数
num_epochs = 100
initial_lr = 1e-3   # 初始学习率
final_lr = 5e-4     # 最终学习率

print("\n开始训练 (三层Pre-LN Transformer + Dropout)...")
best_val_loss = float('inf')
grad_print_flag = False  # 每个epoch仅打印一次梯度详情

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_total_loss = 0.0
    train_total_samples = 0
    
    # 余弦退火学习率 + 预热 (前10%轮次线性增加)
    if epoch < num_epochs * 0.1:
        warmup_factor = epoch / (num_epochs * 0.1)
        lr = initial_lr * warmup_factor
    else:
        # 余弦退火: 从initial_lr平滑衰减到final_lr
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (
            1 + np.cos(np.pi * (epoch - num_epochs * 0.1) / (num_epochs * 0.9))
        )
    
    # 训练阶段 (随机批次顺序)
    train_batch_indices = np.random.permutation(len(train_sample_batches))
    
    for batch_idx in train_batch_indices:
        X_batch = train_sample_batches[batch_idx]  # [B, τ, 4]
        y_true = train_label_batches[batch_idx]    # [B]
        B_actual = X_batch.shape[0]
        
        # ------------------------------------------------------------------
        # 前向传播 (三层Pre-LN架构)
        # 关键张量维度:
        #   X_batch: [B, τ, 4]
        #   E_batch: [B, τ, d_model] - 嵌入层输出
        #   Z_batch: [B, τ, d_model] - 每层输出
        #   final_repr: [B, d_model] - 序列聚合表示
        #   y_pred: [B] - 预测值
        # ------------------------------------------------------------------
        # 1. 嵌入层 + 位置编码
        E_batch = X_batch @ params['W_e'] + params['b_e']  # [B, τ, d_model]
        Z_batch = E_batch + P  # 添加固定位置编码 [B, τ, d_model]
        
        # 保存每层中间变量 (用于反向传播)
        layer_caches = []
        
        # 2. 三层Transformer编码器 (Pre-LN架构)
        for layer_idx in range(num_layers):
            # 获取当前层参数
            W_Q = params[f'layer{layer_idx}_W_Q']
            W_K = params[f'layer{layer_idx}_W_K']
            W_V = params[f'layer{layer_idx}_W_V']
            W_O = params[f'layer{layer_idx}_W_O']
            W_1 = params[f'layer{layer_idx}_W_1']
            b_1 = params[f'layer{layer_idx}_b_1']
            W_2 = params[f'layer{layer_idx}_W_2']
            b_2 = params[f'layer{layer_idx}_b_2']
            gamma1 = params[f'layer{layer_idx}_gamma1']  # MHA前LayerNorm
            beta1 = params[f'layer{layer_idx}_beta1']
            gamma2 = params[f'layer{layer_idx}_gamma2']  # FFN前LayerNorm
            beta2 = params[f'layer{layer_idx}_beta2']
            
            # [Pre-LN] 1. LayerNorm + MHA
            LN_Z, cache_LN1 = LayerNorm_with_grad(Z_batch, gamma1, beta1)  # [B, τ, d_model]
            (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list,
             V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V) = MHA(
                LN_Z, W_Q, W_K, W_V, W_O, h, d_K
            )  # outs_MHA: [B, τ, d_model]
            
            # Dropout after MHA (训练时)
            outs_MHA_dropped, mask_mha = Dropout(outs_MHA, 0.1, training=True)
            
            # [Pre-LN] 2. 残差连接1: Z + Dropout(MHA(LayerNorm(Z)))
            res1 = Z_batch + outs_MHA_dropped  # [B, τ, d_model]
            
            # [Pre-LN] 3. LayerNorm + FFN
            LN_res1, cache_LN2 = LayerNorm_with_grad(res1, gamma2, beta2)  # [B, τ, d_model]
            outs_FFN, L_1, A = FFN(LN_res1, W_1, b_1, W_2, b_2)  # [B, τ, d_model]
            
            # Dropout after FFN (训练时)
            outs_FFN_dropped, mask_ffn = Dropout(outs_FFN, 0.2, training=True)
            
            # [Pre-LN] 4. 残差连接2: res1 + Dropout(FFN(LayerNorm(res1)))
            Z_batch = res1 + outs_FFN_dropped  # [B, τ, d_model]
            
            # 保存中间变量
            layer_caches.append({
                'LN_Z': LN_Z,          # [B, τ, d_model]
                'cache_LN1': cache_LN1,
                'outs_MHA': outs_MHA,  # [B, τ, d_model]
                'mask_mha': mask_mha,  # Dropout掩码
                'res1': res1,          # [B, τ, d_model]
                'LN_res1': LN_res1,    # [B, τ, d_model]
                'cache_LN2': cache_LN2,
                'outs_FFN': outs_FFN,  # [B, τ, d_model]
                'mask_ffn': mask_ffn,  # Dropout掩码
                'L_1': L_1,            # [B, τ, d_ff]
                'A': A,                # [B, τ, d_ff]
                'AWs': AWs,            # 注意力权重列表
                'AS_originals': AS_originals,
                'max_AS_list': max_AS_list,
                'sum_exp_AS_list': sum_exp_AS_list,
                'V_is': V_is,
                'Q_iso': Q_iso,        # [B, h, τ, d_k]
                'K_iso': K_iso,        # [B, h, τ, d_k]
                'V_iso': V_iso,        # [B, h, τ, d_v]
                'concat_out': concat_out,  # [B, τ, d_model]
                'Q': Q,                # [B, τ, d_model]
                'K': K,                # [B, τ, d_model]
                'V': V                 # [B, τ, d_model]
            })
        
        # 3. 回归预测头
        final_repr = np.mean(Z_batch, axis=1)  # [B, d_model] - 平均池化
        y_pred = (final_repr @ params['W_pred'] + params['b_pred']).squeeze(-1)  # [B]
        
        # 计算MSE损失
        loss = np.mean((y_pred - y_true) ** 2)
        train_total_loss += loss * B_actual
        train_total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播 (精确梯度计算)
        # 设计要点:
        #   - 从回归头开始反向传播
        #   - 注意力层梯度分解 (Q/K/V/O)
        #   - LayerNorm梯度解析计算
        #   - Dropout梯度: 应用相同掩码
        #   - 梯度裁剪: 按参数类型不同裁剪阈值
        # ------------------------------------------------------------------
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头梯度
        dL_dy_pred = 2 * (y_pred - y_true) / B_actual  # [B]
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)  # [d_model, 1]
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)  # [1]
        
        # 平均池化梯度: 均匀分配到每个时间步
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ params['W_pred'].T).reshape(B_actual, d_model)  # [B, d_model]
        dL_dlast_layer = np.tile(dL_dfinal_repr[:, np.newaxis, :], (1, tau, 1)) / tau  # [B, τ, d_model]
        
        # 梯度监控 (每epoch第一次)
        if not grad_print_flag:
            print(f"\n【梯度监控-前】W_pred梯度幅值: {np.abs(grads['W_pred']).mean():.6f}")
        
        # 梯度裁剪 (按参数类型)
        for name in grads.keys():
            if 'W_pred' in name or 'b_pred' in name:
                grads[name] = np.clip(grads[name], -0.01, 0.01)  # 回归头严格裁剪
            elif 'layer2' in name:  # 深层中等裁剪
                grads[name] = np.clip(grads[name], -0.05, 0.05)
            else:  # 其他参数
                grads[name] = np.clip(grads[name], -0.1, 0.1)
        
        if not grad_print_flag:
            print(f"【梯度监控-后】W_pred梯度幅值: {np.abs(grads['W_pred']).mean():.6f}")
        
        # 2. 反向传播通过各层 (从深层到浅层)
        dL_dZ = dL_dlast_layer  # [B, τ, d_model]
        for layer_idx in reversed(range(num_layers)):
            cache = layer_caches[layer_idx]
            
            # 获取当前层参数
            W_Q = params[f'layer{layer_idx}_W_Q']
            W_K = params[f'layer{layer_idx}_W_K']
            W_V = params[f'layer{layer_idx}_W_V']
            W_O = params[f'layer{layer_idx}_W_O']
            W_1 = params[f'layer{layer_idx}_W_1']
            W_2 = params[f'layer{layer_idx}_W_2']
            b_1 = params[f'layer{layer_idx}_b_1']
            b_2 = params[f'layer{layer_idx}_b_2']
            gamma1 = params[f'layer{layer_idx}_gamma1']
            beta1 = params[f'layer{layer_idx}_beta1']
            gamma2 = params[f'layer{layer_idx}_gamma2']
            beta2 = params[f'layer{layer_idx}_beta2']
            
            # [Pre-LN] 4. 残差连接2的反向
            dL_dres1 = dL_dZ.copy()       # 直连路径
            dL_douts_FFN = dL_dZ.copy()   # FFN路径
            
            # Dropout梯度 (反向应用相同掩码)
            if cache['mask_ffn'] is not None:
                dL_douts_FFN = dL_douts_FFN * cache['mask_ffn']  # [B, τ, d_model]
            
            # FFN反向
            dL_dL2 = dL_douts_FFN  # [B, τ, d_model]
            # W_2梯度
            grads[f'layer{layer_idx}_W_2'] = cache['A'].reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)  # [d_ff, d_model]
            grads[f'layer{layer_idx}_b_2'] = np.sum(dL_dL2, axis=(0,1))  # [d_model]
            # A梯度
            dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T  # [B*τ, d_ff]
            dL_dA = dL_dA.reshape(B_actual, tau, d_ff)  # [B, τ, d_ff]
            
            # Swish梯度
            sigmoid_L1 = 1.0 / (1.0 + np.exp(-cache['L_1']))
            dSwish_dL1 = sigmoid_L1 * (1 + cache['L_1'] * (1 - sigmoid_L1))  # [B, τ, d_ff]
            dL_dL1 = dL_dA * dSwish_dL1  # [B, τ, d_ff]
            
            # FFN第一层梯度
            grads[f'layer{layer_idx}_W_1'] = cache['LN_res1'].reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)  # [d_model, d_ff]
            grads[f'layer{layer_idx}_b_1'] = np.sum(dL_dL1, axis=(0,1))  # [d_ff]
            
            # [Pre-LN] 3. LayerNorm2的反向
            _, (dL_dres1_from_LN2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
                cache['res1'], gamma2, beta2, dL_dL1 @ W_1.T  # [B, τ, d_model]
            )
            grads[f'layer{layer_idx}_gamma2'] = dL_dgamma2  # [d_model]
            grads[f'layer{layer_idx}_beta2'] = dL_dbeta2    # [d_model]
            dL_dres1 += dL_dres1_from_LN2  # 梯度累加
            
            # [Pre-LN] 2. 残差连接1的反向
            dL_dinput = dL_dres1.copy()     # 直连路径
            dL_douts_MHA = dL_dres1.copy()  # MHA路径
            
            # Dropout梯度
            if cache['mask_mha'] is not None:
                dL_douts_MHA = dL_douts_MHA * cache['mask_mha']  # [B, τ, d_model]
            
            # MHA反向
            grads[f'layer{layer_idx}_W_O'] = cache['concat_out'].reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)  # [d_model, d_model]
            dL_dconcat_out = dL_douts_MHA.reshape(-1, d_model) @ W_O.T  # [B*τ, d_model]
            dL_dconcat_out = dL_dconcat_out.reshape(B_actual, tau, d_model)  # [B, τ, d_model]
            
            # 初始化Q/K/V梯度
            dL_dQ_total = np.zeros((B_actual, tau, d_model))
            dL_dK_total = np.zeros((B_actual, tau, d_model))
            dL_dV_total = np.zeros((B_actual, tau, d_model))
            
            # 逐头反向传播
            for i in range(h):
                dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]  # [B, τ, d_k]
                AW_i = cache['AWs'][i]        # [B, τ, τ]
                V_i = cache['V_is'][i]        # [B, τ, d_v]
                Q_i = cache['Q_iso'][:, i, :, :]  # [B, τ, d_k]
                K_i = cache['K_iso'][:, i, :, :]  # [B, τ, d_k]
                AS_original_i = cache['AS_originals'][i]  # [B, τ, τ]
                max_AS_i = cache['max_AS_list'][i]        # [B, τ, 1]
                sum_exp_AS_i = cache['sum_exp_AS_list'][i]  # [B, τ, 1]
                
                # 1. dL/dV_i = AW_i^T @ dL/dout_i
                dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)  # [B, τ, d_v]
                
                # 2. dL/dAW = dL/dout_i @ V_i^T
                dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))  # [B, τ, τ]
                # 3. dL/dAS = AW_i * (dL_dAW - sum(dL_dAW * AW_i))
                dL_dAS = AW_i * (dL_dAW - np.sum(dL_dAW * AW_i, axis=-1, keepdims=True))  # [B, τ, τ]
                # 4. dL/dmax_AS = sum(dL_dAS)
                dL_dmax_AS = np.sum(dL_dAS, axis=-1, keepdims=True)  # [B, τ, 1]
                # 5. 处理数值稳定化带来的梯度
                mask = (AS_original_i == max_AS_i).astype(np.float32)  # [B, τ, τ]
                mask_sum = np.sum(mask, axis=-1, keepdims=True) + 1e-8  # [B, τ, 1]
                dL_dAS_original = dL_dAS - mask * dL_dmax_AS / mask_sum  # [B, τ, τ]
                
                # 6. dL/dQ_i = (dL_dAS_original / √d_k) @ K_i
                dL_dQ_i = np.matmul(dL_dAS_original / np.sqrt(d_K), K_i)  # [B, τ, d_k]
                # 7. dL/dK_i = (dL_dAS_original^T / √d_k) @ Q_i
                dL_dK_i = np.matmul(dL_dAS_original.transpose(0,2,1) / np.sqrt(d_K), Q_i)  # [B, τ, d_k]
                
                # 累加到总梯度
                dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
                dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
                dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
            
            # Q/K/V投影梯度 (使用LayerNorm输出作为输入)
            grads[f'layer{layer_idx}_W_Q'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)  # [d_model, d_model]
            grads[f'layer{layer_idx}_W_K'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)  # [d_model, d_model]
            grads[f'layer{layer_idx}_W_V'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)  # [d_model, d_model]
            
            # [Pre-LN] 1. LayerNorm1的反向
            dL_dinput_from_LN1, dL_dgamma1, dL_dbeta1 = LayerNorm_with_grad(
                cache['LN_Z'], gamma1, beta1, 
                dL_dQ_total @ W_Q.T + dL_dK_total @ W_K.T + dL_dV_total @ W_V.T
            )[1]
            grads[f'layer{layer_idx}_gamma1'] = dL_dgamma1  # [d_model]
            grads[f'layer{layer_idx}_beta1'] = dL_dbeta1    # [d_model]
            dL_dinput += dL_dinput_from_LN1  # 梯度累加
            
            # 传递到前一层
            dL_dZ = dL_dinput  # [B, τ, d_model]
        
        # 3. 嵌入层梯度
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dZ.reshape(-1, d_model)  # [4, d_model]
        grads['b_e'] = np.sum(dL_dZ, axis=(0,1))  # [d_model]
        
        # ------------------------------------------------------------------
        # 优化器更新
        # ------------------------------------------------------------------
        optimizer.step(grads, lr=lr)
        
        # 梯度调试 (每epoch第一次)
        if not grad_print_flag:
            print(f"\n【调试-梯度幅值】Epoch {epoch+1} 第1个批次梯度统计：")
            print(f"  layer0_W_1梯度均值: {grads['layer0_W_1'].mean():.8f}, 绝对值均值: {np.abs(grads['layer0_W_1']).mean():.8f}")
            print(f"  layer2_W_2梯度均值: {grads['layer2_W_2'].mean():.8f}, 绝对值均值: {np.abs(grads['layer2_W_2']).mean():.8f}")
            print(f"  W_pred梯度均值: {grads['W_pred'].mean():.8f}, 绝对值均值: {np.abs(grads['W_pred']).mean():.8f}")
            print(f"\n【调试-中间变量】layer0输入均值: {layer_caches[0]['LN_Z'].mean():.6f}, 标准差: {layer_caches[0]['LN_Z'].std():.6f}")
            print(f"【调试-中间变量】layer2输出均值: {Z_batch.mean():.6f}, 标准差: {Z_batch.std():.6f}")
            print(f"【调试-中间变量】y_pred均值: {y_pred.mean():.6f}, y_true均值: {y_true.mean():.6f}")
            
            # 梯度健康监控 (深层/浅层梯度幅值比)
            layer0_mha_grad = np.abs(grads['layer0_W_Q']).mean()
            layer2_mha_grad = np.abs(grads['layer2_W_Q']).mean()
            mha_ratio = layer2_mha_grad / (layer0_mha_grad + 1e-8) if layer0_mha_grad > 1e-8 else 0.0
            
            layer0_ffn_grad = np.abs(grads['layer0_W_1']).mean()
            layer2_ffn_grad = np.abs(grads['layer2_W_1']).mean()
            ffn_ratio = layer2_ffn_grad / (layer0_ffn_grad + 1e-8) if layer0_ffn_grad > 1e-8 else 0.0
            
            # Dropout实际比率
            mha_dropout_rate = np.mean(1.0 - cache['mask_mha']) if cache['mask_mha'] is not None else 0.0
            ffn_dropout_rate = np.mean(1.0 - cache['mask_ffn']) if cache['mask_ffn'] is not None else 0.0
            
            print(f"\n【梯度健康】MHA梯度幅值比: {mha_ratio:.4f} (健康范围: 0.5-2.0)")
            print(f"【梯度健康】FFN梯度幅值比: {ffn_ratio:.4f} (健康范围: 0.5-2.0)")
            print(f"【Dropout监控】MHA实际比率: {mha_dropout_rate:.3f}, FFN实际比率: {ffn_dropout_rate:.3f}")
            
            grad_print_flag = True
    
    # ------------------------------------------------------------------
    # 验证阶段 (无Dropout)
    # ------------------------------------------------------------------
    val_total_loss = 0.0
    val_total_samples = 0
    
    with np.errstate(all='ignore'):
        for batch_idx in range(len(val_sample_batches)):
            X_batch = val_sample_batches[batch_idx]  # [B, τ, 4]
            y_true = val_label_batches[batch_idx]    # [B]
            B_actual = X_batch.shape[0]
            
            # 验证前向传播 (无Dropout)
            E_batch = X_batch @ params['W_e'] + params['b_e']  # [B, τ, d_model]
            Z_batch = E_batch + P
            
            # 三层Transformer (相同结构，无Dropout)
            for layer_idx in range(num_layers):
                W_Q = params[f'layer{layer_idx}_W_Q']
                W_K = params[f'layer{layer_idx}_W_K']
                W_V = params[f'layer{layer_idx}_W_V']
                W_O = params[f'layer{layer_idx}_W_O']
                W_1 = params[f'layer{layer_idx}_W_1']
                b_1 = params[f'layer{layer_idx}_b_1']
                W_2 = params[f'layer{layer_idx}_W_2']
                b_2 = params[f'layer{layer_idx}_b_2']
                gamma1 = params[f'layer{layer_idx}_gamma1']
                beta1 = params[f'layer{layer_idx}_beta1']
                gamma2 = params[f'layer{layer_idx}_gamma2']
                beta2 = params[f'layer{layer_idx}_beta2']
                
                # [Pre-LN] 1. LayerNorm + MHA
                LN_Z = LayerNorm(Z_batch, gamma1, beta1)
                outs_MHA, _, _, _, _, _, _, _, _, _, _, _, _, _ = MHA(
                    LN_Z, W_Q, W_K, W_V, W_O, h, d_K
                )
                
                res1 = Z_batch + outs_MHA
                
                # [Pre-LN] 2. LayerNorm + FFN
                LN_res1 = LayerNorm(res1, gamma2, beta2)
                outs_FFN, _, _ = FFN(LN_res1, W_1, b_1, W_2, b_2)
                
                Z_batch = res1 + outs_FFN
            
            final_repr = np.mean(Z_batch, axis=1)  # [B, d_model]
            y_pred = (final_repr @ params['W_pred'] + params['b_pred']).squeeze(-1)  # [B]
            
            loss = np.mean((y_pred - y_true) ** 2)
            val_total_loss += loss * B_actual
            val_total_samples += B_actual
    
    # ------------------------------------------------------------------
    # 结果统计与模型保存
    # ------------------------------------------------------------------
    avg_train_loss = train_total_loss / train_total_samples
    avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else float('inf')
    epoch_time = time.time() - epoch_start_time
    
    # 早停策略：保存验证损失最低的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        # 保存模型 + 归一化统计量 (用于推理)
        np.savez("./model/best_transformer_params.npz",
                 **params, 
                 mean_X_train=mean_X_train, std_X_train=std_X_train,
                 mean_Y_train=mean_Y_train, std_Y_train=std_Y_train)
        print(f"最优验证损失更新: {best_val_loss:.6f}，已保存模型")
    
    # 打印训练信息
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.6f} - "
          f"Val Loss: {avg_val_loss:.6f} - "
          f"Best Val Loss: {best_val_loss:.6f} - "
          f"LR: {lr:.6f} - Time: {epoch_time:.2f}s")
    
    # 重置梯度打印标记
    grad_print_flag = False

print("\n训练完成！")
print(f"最优验证损失: {best_val_loss:.6f}")

# 保存最终模型
np.savez("./model/transformer_params.npz",
         **params,
         mean_X_train=mean_X_train, std_X_train=std_X_train,
         mean_Y_train=mean_Y_train, std_Y_train=std_Y_train)
print("模型已保存到 ./model/，包含三层Pre-LN Transformer + Dropout参数和归一化统计量")