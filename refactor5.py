import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# （1）数据预处理（无修改）
# ------------------------------------------------------------------

# 读取原始数据（未归一化）
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 分离特征和标签（未归一化）
seq_X = seq[:,:4]
seq_Y = seq[:,4]

# 超参数：滑动窗口长度 τ
tau = 16

# 生成未归一化的样本和标签
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(seq_X)-tau):
    samples.append(seq_X[i:i+tau,:])
    labels.append(seq_Y[i+tau])

# 划分训练集和验证集（时序数据不随机划分）
split_idx = int(len(samples) * 0.8)
train_samples_raw = samples[:split_idx]  
train_labels_raw = labels[:split_idx]    
val_samples_raw = samples[split_idx:]    
val_labels_raw = labels[split_idx:]      

# 仅用训练集计算归一化的均值和方差（避免未来信息泄露）
train_X_concat = np.concatenate(train_samples_raw, axis=0)  
train_Y_flat = np.array(train_labels_raw)                  
mean_X, std_X = train_X_concat.mean(axis=0), train_X_concat.std(axis=0)
mean_Y, std_Y = train_Y_flat.mean(), train_Y_flat.std()

# 打印关键验证指标（确认1.0基线）
print(f"标签标准差 std_Y = {std_Y:.4f}（验证1.0基线）")

# 分别归一化训练集和验证集（验证集复用训练集的统计量）
def normalize_data(samples, labels, mean_X, std_X, mean_Y, std_Y):
    norm_samples = []
    norm_labels = []
    for sample, label in zip(samples, labels):
        norm_sample = (sample - mean_X) / (std_X + 1e-8)
        norm_label = (label - mean_Y) / (std_Y + 1e-8)
        norm_samples.append(norm_sample)
        norm_labels.append(norm_label)
    return norm_samples, norm_labels

train_samples, train_labels = normalize_data(train_samples_raw, train_labels_raw, mean_X, std_X, mean_Y, std_Y)
val_samples, val_labels = normalize_data(val_samples_raw, val_labels_raw, mean_X, std_X, mean_Y, std_Y)

print(f"数据集划分完成 - 训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")

# 超参数：批量 B
B = 32

# 生成训练集批次（过滤空批次，避免形状不一致）
train_sample_batches = []  # (B,τ,4)
for i in range(0, len(train_samples), B):
    batch = np.array(train_samples[i:i+B])
    if len(batch) > 0:  # 过滤空批次
        train_sample_batches.append(batch)
train_label_batches = []  # (B,)
for i in range(0, len(train_labels), B):
    batch = np.array(train_labels[i:i+B])
    if len(batch) > 0:  # 过滤空批次
        train_label_batches.append(batch)

# 生成验证集批次（过滤空批次，避免形状不一致）
val_sample_batches = []  # (B,τ,4)
for i in range(0, len(val_samples), B):
    batch = np.array(val_samples[i:i+B])
    if len(batch) > 0:  # 过滤空批次
        val_sample_batches.append(batch)
val_label_batches = []  # (B,)
for i in range(0, len(val_labels), B):
    batch = np.array(val_labels[i:i+B])
    if len(batch) > 0:  # 过滤空批次
        val_label_batches.append(batch)

# ------------------------------------------------------------------
# （2）模型参数初始化（【核心修复】MHA Q/K初始化缩放+LeakyReLU）
# ------------------------------------------------------------------

d_model = 64
d_in = 4

# Kaiming 初始化方法（修正fan_in计算，适配不同层）
def KaimingInit(shape, fan_in):
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)

# 特征嵌入层线性投影的权重和偏置
W_e = KaimingInit((d_in, d_model), d_in)  # (4,d_model)
b_e = np.zeros(d_model)                   # (d_model,)

# 位置编码（基础频率1000）
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(1000.0)/d_model))
P = np.zeros((tau,d_model))  # (τ,d_model)
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

# 注意力头数
h = 8
d_K = d_model // h  # 单头维度=8
d_V = d_K

# 【核心修复1】Q/K矩阵初始化缩小10倍，解决初始熵值异常
W_Q = KaimingInit((d_model, d_model), d_model) * 0.1
W_K = KaimingInit((d_model, d_model), d_model) * 0.1
W_V = KaimingInit((d_model, d_model), d_model)
W_O = KaimingInit((d_model, d_model), d_model)

# 前馈维度
d_ff = 4 * d_model
# 修正FFN W_1的fan_in（d_model→d_model，避免初始化值过小）
W_1 = KaimingInit((d_model, d_ff), d_model)
b_1 = np.zeros(d_ff)
W_2 = KaimingInit((d_ff, d_model), d_ff)
b_2 = np.zeros(d_model)

# LayerNorm 参数
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

# 回归头
W_pred = KaimingInit((d_model,1), d_model)
b_pred = np.array([0.0])

# ------------------------------------------------------------------
# （3）辅助函数（【核心修复】LeakyReLU替换ReLU+梯度裁剪）
# ------------------------------------------------------------------

# 层归一化
def LayerNorm(Z, gamma, beta):
    mean = np.mean(Z,axis=-1,keepdims=True)
    std = np.std(Z,axis=-1,keepdims=True)
    return gamma*((Z-mean)/(std+1e-8))+beta

# 带梯度的层归一化
def LayerNorm_with_grad(Z, gamma, beta, dL_dout=None):
    mean = np.mean(Z, axis=-1, keepdims=True)
    std = np.std(Z, axis=-1, keepdims=True)
    norm_Z = (Z - mean) / (std + 1e-8)
    out = gamma * norm_Z + beta
    
    if dL_dout is None:
        return out, None
    
    B, T, D = Z.shape
    dL_dgamma = np.sum(dL_dout * norm_Z, axis=(0,1))
    dL_dbeta = np.sum(dL_dout, axis=(0,1))
    
    dL_dnorm = dL_dout * gamma
    dL_dstd = np.sum(dL_dnorm * (Z - mean) * -0.5 * (std + 1e-8)**-3, axis=-1, keepdims=True)
    dL_dmean = np.sum(dL_dnorm * -1.0 / (std + 1e-8), axis=-1, keepdims=True)
    dL_dmean += dL_dstd * np.mean(-2.0 * (Z - mean), axis=-1, keepdims=True)
    
    dL_dZ = dL_dnorm / (std + 1e-8)
    dL_dZ += dL_dstd * 2.0 * (Z - mean) / D
    dL_dZ += dL_dmean / D
    
    return out, (dL_dZ, dL_dgamma, dL_dbeta)

# 缩放点积注意力（修正d_K缩放+数值稳定性）
def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    # Q_i/K_i/V_i: (B, τ, d_K)
    B, T, _ = Q_i.shape
    
    # 确保注意力得分计算维度正确，缩放因子为sqrt(d_K)
    AS_original = np.matmul(Q_i, K_i.transpose(0,2,1))  # (B, τ, τ)
    scale = np.sqrt(d_K)
    AS_original = AS_original / (scale + 1e-8)  # 避免除零
    
    # 增强数值稳定性，防止exp溢出
    max_AS = np.max(AS_original, axis=-1, keepdims=True)
    AS = AS_original - max_AS  # 平移到负数区间
    exp_AS = np.exp(AS)
    sum_exp_AS = np.sum(exp_AS, axis=-1, keepdims=True) + 1e-8  # 避免除零
    AW = exp_AS / sum_exp_AS
    
    # 单头注意力输出
    out = np.matmul(AW, V_i)  # (B, τ, d_K)
    
    # 返回所有中间变量用于反向传播
    return out, AW, AS_original, AS, max_AS, sum_exp_AS

# 多头注意力实现（修正Q/K/V维度转换）
def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    B, tau, d_model = Z.shape
    
    # 计算 Q、K、V，(B,τ,d_model)
    Q = np.matmul(Z, W_Q)
    K = np.matmul(Z, W_K)
    V = np.matmul(Z, W_V)
    
    # 修正Q/K/V的reshape和transpose维度（避免异常聚焦）
    # 拆分到h个头：(B, τ, h, d_K) → (B, h, τ, d_K)
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    
    outs = []
    AWs = []
    AS_originals = []
    AS_list = []
    max_AS_list = []
    sum_exp_AS_list = []
    V_is = []
    
    # 计算单头注意力
    for i in range(h):
        Q_i = Q_iso[:, i, :, :]  # (B, τ, d_K)
        K_i = K_iso[:, i, :, :]  # (B, τ, d_K)
        V_i = V_iso[:, i, :, :]  # (B, τ, d_K)
        
        out, AW, AS_original, AS, max_AS, sum_exp_AS = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)
        AWs.append(AW)
        AS_originals.append(AS_original)
        AS_list.append(AS)
        max_AS_list.append(max_AS)
        sum_exp_AS_list.append(sum_exp_AS)
        V_is.append(V_i)
    
    # 拼接多头输出：(B, τ, h*d_K) = (B, τ, d_model)
    concat_out = np.concatenate(outs, axis=-1)
    # 输出投影
    outs_MHA = np.matmul(concat_out, W_O)
    
    return (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list, 
            V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V)

# 【核心修复2】LeakyReLU替换ReLU，避免梯度全0
def LeakyReLU(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# 【核心修复3】梯度裁剪函数，避免梯度爆炸
def clip_gradient(grads, max_norm=1.0):
    clipped_grads = {}
    total_norm = 0.0
    for name, grad in grads.items():
        norm = np.linalg.norm(grad)
        total_norm += norm ** 2
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-8)
    if clip_coef < 1.0:
        for name, grad in grads.items():
            clipped_grads[name] = grad * clip_coef
    else:
        clipped_grads = grads
    return clipped_grads

# 前馈网络实现（修正梯度计算逻辑）
def FFN(Z, W_1, b_1, W_2, b_2):
    # Z: (B, τ, d_model)
    B, τ, d_model = Z.shape
    
    # 第一层线性变换：(B, τ, d_ff)
    L_1 = np.matmul(Z, W_1) + b_1
    # LeakyReLU激活（替换ReLU）
    A = LeakyReLU(L_1)
    # 第二层线性变换：(B, τ, d_model)
    L_2 = np.matmul(A, W_2) + b_2
    
    return L_2, L_1, A

# ------------------------------------------------------------------
# （4）AdamW 优化器（保持不变）
# ------------------------------------------------------------------

class AdamWOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}
        self.t = 0
        
        self.weight_params = ['W_e', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2', 'W_pred']
    
    def step(self, grads, lr=None):
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        
        for name in self.params.keys():
            param = self.params[name]
            grad = grads[name]
            
            # 更新矩估计
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # AdamW 更新
            if name in self.weight_params:
                update = current_lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)
            else:
                update = current_lr * (m_hat / (np.sqrt(v_hat) + self.eps))
            
            self.params[name] = param - update
    
    def state_dict(self):
        return {'m': self.m, 'v': self.v, 't': self.t}
    
    def load_state_dict(self, state_dict):
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']

# ------------------------------------------------------------------
# （5）训练循环（【核心修复】回归头改最后一步+梯度裁剪）
# ------------------------------------------------------------------

params = {
    'W_e': W_e,
    'b_e': b_e,
    'W_Q': W_Q,
    'W_K': W_K,
    'W_V': W_V,
    'W_O': W_O,
    'W_1': W_1,
    'b_1': b_1,
    'W_2': W_2,
    'b_2': b_2,
    'gamma1': gamma1,
    'beta1': beta1,
    'gamma2': gamma2,
    'beta2': beta2,
    'W_pred': W_pred,
    'b_pred': b_pred
}

# 调整学习率和权重衰减，减缓过拟合
optimizer = AdamWOptimizer(
    params,
    lr=3e-4,  # 降低学习率
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=5e-5  # 降低权重衰减
)

num_epochs = 100
initial_lr = 3e-4  # 同步降低初始学习率
final_lr = 1e-5

print("开始训练...")
best_val_loss = float('inf')

# 用于验证注意力熵值的标记（仅打印一次）
print_attention_entropy = True
# 用于调试梯度的标记（仅第5轮第0批次打印）
debug_grad = False

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_total_loss = 0.0
    train_total_samples = 0
    
    # 余弦退火学习率调度
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / num_epochs))
    
    # 训练阶段
    train_batch_indices = np.random.permutation(len(train_sample_batches))
    
    for batch_idx in train_batch_indices:
        X_batch = train_sample_batches[batch_idx]
        y_true = train_label_batches[batch_idx]
        B_actual = X_batch.shape[0]
        
        W_e, b_e = params['W_e'], params['b_e']
        W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
        W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
        gamma1, beta1 = params['gamma1'], params['beta1']
        gamma2, beta2 = params['gamma2'], params['beta2']
        W_pred, b_pred = params['W_pred'], params['b_pred']
        
        # 前向传播
        E_batch = X_batch @ W_e + b_e
        Z_batch = E_batch + P
        
        (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list,
         V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V) = MHA(
            Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        
        # 验证注意力熵值（仅第1轮打印）
        if print_attention_entropy and epoch == 0:
            AW = AWs[0][0]  # 第一个样本第一个头的注意力矩阵
            entropy = -np.sum(AW * np.log(AW + 1e-8), axis=-1).mean()
            print(f"初始注意力熵值 = {entropy:.4f}（均匀分布熵值≈{np.log(16):.4f}）")
            print_attention_entropy = False
        
        res_1 = Z_batch + outs_MHA
        outs_LN_1 = LayerNorm(res_1, gamma1, beta1)
        
        outs_FFN, L_1, A = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
        
        res_2 = outs_LN_1 + outs_FFN
        outs_LN_2 = LayerNorm(res_2, gamma2, beta2)
        
        # 【核心修复4】回归头从全局池化改为取最后一步（单步预测核心修正）
        # 原代码：final_repr = np.mean(outs_LN_2, axis=1)
        final_repr = outs_LN_2[:, -1, :]  # 取最后一个时间步，维度(B, d_model)
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
        
        loss = np.mean((y_pred - y_true) ** 2)
        train_total_loss += loss * B_actual
        train_total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播（核心：FFN梯度修复 + 全链路调试打印）
        # ------------------------------------------------------------------
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头梯度（适配最后一步采样）
        dL_dy_pred = 2 * (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)
        
        # 【核心修复5】梯度传递适配最后一步采样（无稀释）
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        dL_douts_LN2 = np.zeros_like(outs_LN_2)  # (B, τ, d_model)
        # 仅最后一步有梯度，无稀释
        dL_douts_LN2[:, -1, :] = dL_dfinal_repr
        
        # 【调试打印1】LayerNorm2输入梯度
        if debug_grad:
            print(f"\n===== 梯度调试 - LayerNorm2环节 =====")
            print(f"dL_douts_LN2 均值 = {np.mean(dL_douts_LN2):.8f}, 非零占比 = {np.mean((dL_douts_LN2 != 0).astype(float)):.4f}")
            print(f"dL_douts_LN2 最大值 = {np.max(dL_douts_LN2):.8f}, 最小值 = {np.min(dL_douts_LN2):.8f}")
        
        # 2. LayerNorm2 反向
        _, (dL_dres2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_2, gamma2, beta2, dL_douts_LN2
        )
        grads['gamma2'] = dL_dgamma2
        grads['beta2'] = dL_dbeta2
        
        # 【调试打印2】LayerNorm2反向输出（残差梯度）
        if debug_grad:
            print(f"\n===== 梯度调试 - 残差梯度环节 =====")
            print(f"dL_dres2 均值 = {np.mean(dL_dres2):.8f}, 非零占比 = {np.mean((dL_dres2 != 0).astype(float)):.4f}")
            print(f"dL_dres2 最大值 = {np.max(dL_dres2):.8f}, 最小值 = {np.min(dL_dres2):.8f}")
        
        # 3. FFN + 第二残差梯度（FFN梯度核心修复）
        dL_douts_LN1 = dL_dres2.copy()  # 残差→LN1输出
        dL_douts_FFN = dL_dres2.copy()  # 残差→FFN输出
        
        # 【调试打印3】FFN第二层输入梯度
        if debug_grad:
            print(f"\n===== 梯度调试 - FFN第二层环节 =====")
            print(f"dL_douts_FFN (dL_dL2) 均值 = {np.mean(dL_douts_FFN):.8f}, 非零占比 = {np.mean((dL_douts_FFN != 0).astype(float)):.4f}")
            print(f"dL_douts_FFN 最大值 = {np.max(dL_douts_FFN):.8f}, 最小值 = {np.min(dL_douts_FFN):.8f}")
        
        # FFN第二层反向
        dL_dL2 = dL_douts_FFN  # (B, τ, d_model)
        # 修正W_2梯度计算维度
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        
        # 【调试打印4】W_2梯度
        if debug_grad:
            print(f"W_2 梯度均值 = {np.mean(grads['W_2']):.8f}, 非零占比 = {np.mean((grads['W_2'] != 0).astype(float)):.4f}")
        
        # FFN激活层反向（适配LeakyReLU）
        dL_dA = dL_dL2 @ W_2.T  # (B, τ, d_ff)
        
        # 【调试打印5】激活层梯度
        if debug_grad:
            print(f"\n===== 梯度调试 - FFN激活层环节 =====")
            print(f"dL_dA 均值 = {np.mean(dL_dA):.8f}, 非零占比 = {np.mean((dL_dA != 0).astype(float)):.4f}")
            print(f"dL_dA 最大值 = {np.max(dL_dA):.8f}, 最小值 = {np.min(dL_dA):.8f}")
        
        # LeakyReLU梯度计算（替代ReLU）
        alpha = 0.1
        dLeakyReLU_dL1 = np.ones_like(L_1)
        dLeakyReLU_dL1[L_1 <= 0] = alpha
        
        dL_dL1 = dL_dA * dLeakyReLU_dL1  # (B, τ, d_ff)
        
        # 【调试打印6】FFN第一层梯度
        if debug_grad:
            print(f"\n===== 梯度调试 - FFN第一层环节 =====")
            print(f"dLeakyReLU_dL1 非零占比 = {np.mean((dLeakyReLU_dL1 != 0).astype(float)):.4f}")
            print(f"dL_dL1 均值 = {np.mean(dL_dL1):.8f}, 非零占比 = {np.mean((dL_dL1 != 0).astype(float)):.4f}")
            print(f"dL_dL1 最大值 = {np.max(dL_dL1):.8f}, 最小值 = {np.min(dL_dL1):.8f}")
        
        # 修正FFN W_1梯度计算（核心！解决梯度为0问题）
        # W_1: (d_model, d_ff) = (LN1输出)^T @ dL_dL1
        grads['W_1'] = outs_LN_1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))
        
        # 【调试打印7】W_1梯度计算的输入矩阵
        if debug_grad:
            print(f"\n===== 梯度调试 - W_1梯度计算 =====")
            print(f"outs_LN_1 形状 = {outs_LN_1.shape}, 均值 = {np.mean(outs_LN_1):.8f}")
            print(f"dL_dL1 形状 = {dL_dL1.shape}, 均值 = {np.mean(dL_dL1):.8f}")
            print(f"W_1 梯度形状 = {grads['W_1'].shape}, 均值 = {np.mean(grads['W_1']):.8f}")
            print(f"W_1 梯度非零占比 = {np.mean((grads['W_1'] != 0).astype(float)):.4f}")
        
        # 合并FFN梯度到LN1输出
        dL_douts_LN1_from_FFN = dL_dL1 @ W_1.T  # (B, τ, d_model)
        dL_douts_LN1 += dL_douts_LN1_from_FFN
        
        # 4. LayerNorm1 反向
        _, (dL_dres1, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            res_1, gamma1, beta1, dL_douts_LN1
        )
        grads['gamma1'] = dL_dgamma1
        grads['beta1'] = dL_dbeta1
        
        # 5. 第一残差梯度（双向完整传递）
        dL_dZ_batch = dL_dres1.copy()  # 残差→Z_batch
        dL_douts_MHA = dL_dres1.copy() # 残差→MHA输出
        
        # 6. MHA 反向
        grads['W_O'] = concat_out.reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)
        dL_dconcat_out = dL_douts_MHA @ W_O.T  # (B, τ, d_model)
        
        dL_dQ_total = np.zeros_like(Q)  # (B, τ, d_model)
        dL_dK_total = np.zeros_like(K)  # (B, τ, d_model)
        dL_dV_total = np.zeros_like(V)  # (B, τ, d_model)
        
        for i in range(h):
            # 取出当前头的梯度和中间变量
            dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]  # (B, τ, d_K)
            AW_i = AWs[i]  # (B, τ, τ)
            V_i = V_is[i]  # (B, τ, d_K)
            Q_i = Q_iso[:, i, :, :]  # (B, τ, d_K)
            K_i = K_iso[:, i, :, :]  # (B, τ, d_K)
            AS_original_i = AS_originals[i]  # (B, τ, τ)
            AS_i = AS_list[i]  # (B, τ, τ)
            max_AS_i = max_AS_list[i]  # (B, τ, 1)
            sum_exp_AS_i = sum_exp_AS_list[i]  # (B, τ, 1)
            
            # 计算 dL_dV_i
            dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)  # (B, τ, d_K)
            
            # 计算 dL_dAW_i
            dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))  # (B, τ, τ)
            
            # 计算 dL_dAS_i
            dL_dAS = AW_i * (dL_dAW - np.sum(dL_dAW * AW_i, axis=-1, keepdims=True))  # (B, τ, τ)
            
            # 计算 dL_dAS_original_i
            dL_dmax_AS = np.sum(dL_dAS, axis=-1, keepdims=True)  # (B, τ, 1)
            mask = (AS_original_i == max_AS_i).astype(np.float32)  # (B, τ, τ)
            mask_sum = np.sum(mask, axis=-1, keepdims=True) + 1e-8  # 避免除零
            dL_dAS_original = dL_dAS - mask * dL_dmax_AS / mask_sum  # (B, τ, τ)
            
            # 计算 Q/K 梯度
            dL_dQ_i = np.matmul(dL_dAS_original / (np.sqrt(d_K) + 1e-8), K_i)  # (B, τ, d_K)
            dL_dK_i = np.matmul(dL_dAS_original.transpose(0,2,1) / (np.sqrt(d_K) + 1e-8), Q_i)  # (B, τ, d_K)
            
            # 累加梯度到对应头的位置
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
        
        # Q/K/V 投影梯度
        grads['W_Q'] = Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # MHA梯度合并到Z_batch
        dL_dZ_from_Q = dL_dQ_total @ W_Q.T
        dL_dZ_from_K = dL_dK_total @ W_K.T
        dL_dZ_from_V = dL_dV_total @ W_V.T
        dL_dZ_batch += dL_dZ_from_Q + dL_dZ_from_K + dL_dZ_from_V
        
        # 7. 嵌入层梯度
        dL_dE_batch = dL_dZ_batch
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dE_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dE_batch, axis=(0,1))
        
        # 【核心修复6】梯度裁剪，避免梯度爆炸
        grads = clip_gradient(grads, max_norm=1.0)
        
        # 验证梯度（第5轮打印）
        if epoch == 4 and batch_idx == 0:
            print(f"\n===== 第5轮梯度验证 =====")
            print(f"嵌入层W_e梯度均值 = {np.mean(grads['W_e']):.6f}")
            print(f"FFN W_1梯度均值 = {np.mean(grads['W_1']):.6f}（非0则修复成功）")
            print(f"FFN W_2梯度均值 = {np.mean(grads['W_2']):.6f}")
            print(f"LeakyReLU梯度非零占比 = {np.mean((dLeakyReLU_dL1 != 0).astype(np.float32)):.4f}")
            # 开启调试模式，打印全链路梯度
            debug_grad = True
        
        # 优化器更新
        optimizer.step(grads, lr=lr)
        # 关闭调试模式（仅打印一次）
        debug_grad = False
    
    # 验证阶段
    val_total_loss = 0.0
    val_total_samples = 0
    
    with np.errstate(all='ignore'):
        for batch_idx in range(len(val_sample_batches)):
            X_batch = val_sample_batches[batch_idx]
            y_true = val_label_batches[batch_idx]
            B_actual = X_batch.shape[0]
            
            W_e, b_e = params['W_e'], params['b_e']
            W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
            W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
            gamma1, beta1 = params['gamma1'], params['beta1']
            gamma2, beta2 = params['gamma2'], params['beta2']
            W_pred, b_pred = params['W_pred'], params['b_pred']
            
            E_batch = X_batch @ W_e + b_e
            Z_batch = E_batch + P
            
            outs_MHA, _, _, _, _, _, _, _, _, _, _, _, _, _ = MHA(
                Z_batch, W_Q, W_K, W_V, W_O, h, d_K
            )
            
            res_1 = Z_batch + outs_MHA
            outs_LN_1 = LayerNorm(res_1, gamma1, beta1)
            
            outs_FFN, _, _ = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
            
            res_2 = outs_LN_1 + outs_FFN
            outs_LN_2 = LayerNorm(res_2, gamma2, beta2)
            
            # 验证阶段同样改为最后一步采样
            final_repr = outs_LN_2[:, -1, :]
            y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
            
            loss = np.mean((y_pred - y_true) ** 2)
            val_total_loss += loss * B_actual
            val_total_samples += B_actual
    
    # 结果统计
    avg_train_loss = train_total_loss / train_total_samples
    avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else float('inf')
    epoch_time = time.time() - epoch_start_time
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        np.savez("./model/best_transformer_params.npz",
                 **params, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)
        print(f"✅ 最优验证损失更新: {best_val_loss:.6f}，已保存模型")
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.6f} - "
          f"Val Loss: {avg_val_loss:.6f} - "
          f"Best Val Loss: {best_val_loss:.6f} - "
          f"LR: {lr:.6f} - Time: {epoch_time:.2f}s")

print("训练完成！")
print(f"最优验证损失: {best_val_loss:.6f}")

np.savez("./model/transformer_params.npz",
         **params, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)
print("模型已保存到 ./model/")