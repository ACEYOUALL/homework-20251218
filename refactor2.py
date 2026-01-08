import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# （1）数据预处理
# ------------------------------------------------------------------

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
tau = 5

# 样本和标签
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(norm_seq_X)-tau):
    samples.append(norm_seq_X[i:i+tau,:])
    labels.append(norm_seq_Y[i+tau])

# 超参数：批量 B
B = 8

# 准备样本和标签批次
sample_batches = []  # (B,τ,4)
for i in range(0,len(samples),B):
    sample_batches.append(np.array(samples[i:i+B]))  # 转为 array
label_batches = []  # (B,)
for i in range(0,len(labels),B):
    label_batches.append(np.array(labels[i:i+B]))    # 转为 array

# ------------------------------------------------------------------
# （2）模型参数初始化
# ------------------------------------------------------------------

# 超参数：嵌入维度 d_model
d_model = 64

# 超参数：输入维度 d_in
d_in = 4

# Xavier 初始化权重和偏置
W_e = np.random.randn(d_in,d_model)*np.sqrt(1.0/d_in)  # (4,d_model)
b_e = np.zeros(d_model)                                # (d_model,)

# 位置编码（常量，无需学习）
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))  # (τ,d_model)
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

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

# 超参数：中间层维度
d_ff = 8 * d_model

# Xavier 初始化权重
W_1 = np.random.randn(d_model,d_ff)*np.sqrt(1.0/d_model)
b_1 = np.zeros(d_ff)
W_2 = np.random.randn(d_ff,d_model)*np.sqrt(1.0/d_ff)
b_2 = np.zeros(d_model)

# 层归一化参数
gamma = np.ones(d_model)
beta = np.zeros(d_model)

# Xavier 初始化回归头权重和偏置
W_pred = np.random.randn(d_model,1)*np.sqrt(1.0/d_model)
b_pred = np.zeros(1)

# ------------------------------------------------------------------
# （3）辅助函数定义
# ------------------------------------------------------------------

# 层归一化
def LayerNorm(Z, gamma, beta):
    mean = np.mean(Z,axis=-1,keepdims=True)
    std = np.std(Z,axis=-1,keepdims=True)
    return gamma*((Z-mean)/(std+1e-8))+beta

def LayerNorm_with_grad(Z, gamma, beta, dL_dout=None):
    """前向+反向传播，返回输出和梯度"""
    mean = np.mean(Z, axis=-1, keepdims=True)
    std = np.std(Z, axis=-1, keepdims=True)
    norm_Z = (Z - mean) / (std + 1e-8)
    out = gamma * norm_Z + beta
    
    if dL_dout is None:
        return out, None
    
    # 正确反向传播
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

# 缩放点积注意力
def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    # 注意这里 Q_i、K_i、V_i 是单头，(B,τ,d_K)
    # 注意力得分：QK
    AS = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)
    # softmax（数值稳定形式）计算注意力权重，(B,τ,τ)
    AS = AS - np.max(AS, axis=-1, keepdims=True)
    exp_AS = np.exp(AS)
    AW = exp_AS / np.sum(exp_AS, axis=-1, keepdims=True)
    # 单头输出，(B,τ,d_V)
    out = np.matmul(AW, V_i)
    return out, AW, AS, AW  # 返回额外中间变量用于反向传播

# 多头注意力机制 MHA
def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    B, tau, _ = Z.shape
    # 计算 Q、K、V，(B,τ,d_model)
    Q = np.matmul(Z, W_Q)
    K = np.matmul(Z, W_K)
    V = np.matmul(Z, W_V)
    # 分离 Q、K、V
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0,2,1,3)  # d_V = d_K
    # 注意力结果，(B,τ,d_V)
    outs = []
    # 注意力权重，(B,τ,τ)
    AWs = []
    ASs = []
    V_is = []
    # 计算单头注意力
    for i in range(h):
        Q_i = Q_iso[:,i,:,:]
        K_i = K_iso[:,i,:,:]
        V_i = V_iso[:,i,:,:]
        out, AW, AS, _ = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)
        AWs.append(AW)
        ASs.append(AS)
        V_is.append(V_i)
    # 拼接并得到多头结果，(B,τ,d_V·h) → (B,τ,d_model)
    concat_out = np.concatenate(outs, axis=-1)
    outs_MHA = np.matmul(concat_out, W_O)
    return outs_MHA, AWs, ASs, V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V

# Swish 激活函数
def Swish(x, beta=1.0):
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x*sigmoid

# 前馈网络 FFN
def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1) + b_1
    A = Swish(L_1)
    L_2 = np.matmul(A, W_2) + b_2
    return L_2, L_1, A  # 返回中间变量用于反向传播

# ------------------------------------------------------------------
# （4）训练循环（AdamW 优化器）
# ------------------------------------------------------------------

# 超参数
num_epochs = 30
initial_lr = 0.001  # AdamW 通常用较小学习率
final_lr = 1e-6     # 最终学习率
weight_decay = 1e-2 # AdamW 的权重衰减
beta1 = 0.9         # 一阶矩估计衰减
beta2 = 0.999       # 二阶矩估计衰减
epsilon = 1e-8      # 数值稳定项
clip_value = 10.0    # 梯度裁剪阈值

# AdamW 状态初始化（一阶矩 m，二阶矩 v）
m = {
    'W_e': np.zeros_like(W_e),
    'b_e': np.zeros_like(b_e),
    'W_Q': np.zeros_like(W_Q),
    'W_K': np.zeros_like(W_K),
    'W_V': np.zeros_like(W_V),
    'W_O': np.zeros_like(W_O),
    'W_1': np.zeros_like(W_1),
    'b_1': np.zeros_like(b_1),
    'W_2': np.zeros_like(W_2),
    'b_2': np.zeros_like(b_2),
    'gamma': np.zeros_like(gamma),
    'beta': np.zeros_like(beta),
    'W_pred': np.zeros_like(W_pred),
    'b_pred': np.zeros_like(b_pred)
}

v = {
    'W_e': np.zeros_like(W_e),
    'b_e': np.zeros_like(b_e),
    'W_Q': np.zeros_like(W_Q),
    'W_K': np.zeros_like(W_K),
    'W_V': np.zeros_like(W_V),
    'W_O': np.zeros_like(W_O),
    'W_1': np.zeros_like(W_1),
    'b_1': np.zeros_like(b_1),
    'W_2': np.zeros_like(W_2),
    'b_2': np.zeros_like(b_2),
    'gamma': np.zeros_like(gamma),
    'beta': np.zeros_like(beta),
    'W_pred': np.zeros_like(W_pred),
    'b_pred': np.zeros_like(b_pred)
}

# 时间步（全局）
t = 0

# 权重衰减参数列表（只对权重应用）
weight_params = ['W_e', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2', 'W_pred']
param_dict = {
    'W_e': W_e, 'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O,
    'W_1': W_1, 'W_2': W_2, 'W_pred': W_pred
}

print("开始训练 (AdamW 优化器)...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    total_loss = 0.0
    total_samples = 0
    
    # 余弦退火学习率调度
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / num_epochs))
    
    # 随机打乱批次顺序
    batch_indices = np.random.permutation(len(sample_batches))
    
    for batch_idx in batch_indices:
        t += 1  # 全局时间步增加
        X_batch = sample_batches[batch_idx]  # (B, τ, 4)
        y_true = label_batches[batch_idx]   # (B,)
        B_actual = X_batch.shape[0]
        
        # ------------------------------------------------------------------
        # 前向传播
        # ------------------------------------------------------------------
        
        # 线性投影
        E_batch = X_batch @ W_e + b_e  # (B, τ, d_model)
        
        # 注入位置编码
        Z_batch = E_batch + P  # 广播，(B, τ, d_model)
        
        # 多头注意力
        outs_MHA, AWs, ASs, V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V = MHA(
            Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        
        # 第一次残差连接 + 层归一化
        res_1 = Z_batch + outs_MHA
        outs_LN_1 = LayerNorm(res_1, gamma, beta)
        
        # 前馈网络
        outs_FFN, L_1, A = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
        
        # 第二次残差连接 + 层归一化
        res_2 = outs_LN_1 + outs_FFN
        outs_LN_2 = LayerNorm(res_2, gamma, beta)
        
        # 取最后一步 + 回归头
        final_repr = outs_LN_2[:, -1, :]  # (B, d_model)
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)  # (B,)
        
        # 计算损失（MSE）
        loss = np.mean((y_pred - y_true) ** 2)
        total_loss += loss * B_actual
        total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播（计算梯度）
        # ------------------------------------------------------------------
        
        # 初始化当前批次的梯度
        grads = {
            'W_e': np.zeros_like(W_e),
            'b_e': np.zeros_like(b_e),
            'W_Q': np.zeros_like(W_Q),
            'W_K': np.zeros_like(W_K),
            'W_V': np.zeros_like(W_V),
            'W_O': np.zeros_like(W_O),
            'W_1': np.zeros_like(W_1),
            'b_1': np.zeros_like(b_1),
            'W_2': np.zeros_like(W_2),
            'b_2': np.zeros_like(b_2),
            'gamma': np.zeros_like(gamma),
            'beta': np.zeros_like(beta),
            'W_pred': np.zeros_like(W_pred),
            'b_pred': np.zeros_like(b_pred)
        }
        
        # 1. 回归头
        dL_dy_pred = (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred)
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        
        # 2. LayerNorm2 反向传播
        dL_douts_LN2 = np.zeros_like(outs_LN_2)
        dL_douts_LN2[:, -1, :] = dL_dfinal_repr
        _, (dL_dres2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_2, gamma, beta, dL_douts_LN2
        )
        grads['gamma'] += dL_dgamma2
        grads['beta'] += dL_dbeta2
        
        # 3. 残差 + FFN 反向传播
        dL_douts_LN1 = dL_dres2.copy()
        dL_douts_FFN = dL_dres2.copy()
        
        # FFN 反向传播
        # (a) 第二层
        dL_dL2 = dL_douts_FFN
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        
        dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T
        dL_dA = dL_dA.reshape(B_actual, tau, d_ff)
        
        # (b) Swish 激活函数
        sigmoid_L1 = 1.0 / (1.0 + np.exp(-L_1))
        dSwish_dL1 = sigmoid_L1 * (1 + L_1 * (1 - sigmoid_L1))
        dL_dL1 = dL_dA * dSwish_dL1
        
        # (c) 第一层
        grads['W_1'] = outs_LN_1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))

        # 修复：正确重塑梯度
        dL_dW1T = (dL_dL1.reshape(-1, d_ff) @ W_1.T).reshape(B_actual, tau, d_model)
        dL_douts_LN1 += dL_dW1T
        
        # 4. LayerNorm1 反向传播
        _, (dL_dres1, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            res_1, gamma, beta, dL_douts_LN1
        )
        grads['gamma'] += dL_dgamma1
        grads['beta'] += dL_dbeta1
        
        # 7. 残差连接1 反向传播
        dL_dZ_batch = dL_dres1.copy()
        dL_douts_MHA = dL_dres1.copy()
        
        # 8. MHA 反向传播
        # (a) W_O 梯度
        grads['W_O'] = concat_out.reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)

        # (b) 拼接输出的梯度
        dL_dconcat_out = dL_douts_MHA.reshape(-1, d_model) @ W_O.T
        dL_dconcat_out = dL_dconcat_out.reshape(B_actual, tau, d_model)

        # (c) 初始化总梯度
        dL_dQ_total = np.zeros((B_actual, tau, d_model))
        dL_dK_total = np.zeros((B_actual, tau, d_model))
        dL_dV_total = np.zeros((B_actual, tau, d_model))

        # (d) 对每个头计算梯度
        for i in range(h):
            dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]  # (B, τ, d_K)
            AW_i = AWs[i]             # (B, τ, τ)
            AS_i = ASs[i]             # (B, τ, τ)
            V_i = V_is[i]             # (B, τ, d_K)
            Q_i = Q_iso[:, i, :, :]   # (B, τ, d_K)
            K_i = K_iso[:, i, :, :]   # (B, τ, d_K)
            
            # (i) V 梯度
            dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)  # (B, τ, d_K)
            
            # (ii) AW 梯度
            dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))  # (B, τ, τ)
            
            # (iii) AS 梯度 (softmax 完整导数)
            sum_term = np.sum(dL_dAW * AW_i, axis=-1, keepdims=True)  # (B, τ, 1)
            dL_dAS = AW_i * (dL_dAW - sum_term)  # (B, τ, τ)
            
            # (iv) Q, K 梯度
            dL_dQ_i = np.matmul(dL_dAS, K_i) / np.sqrt(d_K)  # (B, τ, d_K)
            dL_dK_i = np.matmul(dL_dAS.transpose(0,2,1), Q_i) / np.sqrt(d_K)  # (B, τ, d_K)
            
            # (v) 累加到总梯度
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i

        # (e) Q, K, V 的投影梯度
        grads['W_Q'] = Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # 9. Embedding 层梯度
        dL_dE_batch = dL_dZ_batch  # (B, τ, d_model)
        
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dE_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dE_batch, axis=(0,1))
        
        # ------------------------------------------------------------------
        # 优化（AdamW）
        # ------------------------------------------------------------------
        
        # 安全的梯度裁剪函数
        def safe_clip(grad, min_val, max_val):
            if isinstance(grad, np.ndarray):
                if grad.size > 1:  # 非标量数组
                    return np.clip(grad, min_val, max_val)
                else:  # 标量数组
                    return np.array(np.clip(grad.item(), min_val, max_val))
            else:  # Python 标量
                return np.clip(grad, min_val, max_val)
        
        # 应用梯度裁剪
        for param_name, grad in grads.items():
            grads[param_name] = safe_clip(grad, -clip_value, clip_value)
        
        # AdamW 参数更新
        for param_name in grads.keys():
            g = grads[param_name]
            param_old = eval(param_name)
            
            # 更新一阶矩和二阶矩
            m[param_name] = beta1 * m[param_name] + (1 - beta1) * g
            v[param_name] = beta2 * v[param_name] + (1 - beta2) * (g ** 2)
            
            # 偏差修正
            m_hat = m[param_name] / (1 - beta1 ** t)
            v_hat = v[param_name] / (1 - beta2 ** t)
            
            # AdamW 更新规则
            if param_name in weight_params:
                # 解耦权重衰减
                update = lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * param_old)
            else:
                # 无权重衰减
                update = lr * (m_hat / (np.sqrt(v_hat) + epsilon))
            
            # 应用更新
            if param_name == 'W_e':
                W_e -= update
            elif param_name == 'b_e':
                b_e -= update
            elif param_name == 'W_Q':
                W_Q -= update
            elif param_name == 'W_K':
                W_K -= update
            elif param_name == 'W_V':
                W_V -= update
            elif param_name == 'W_O':
                W_O -= update
            elif param_name == 'W_1':
                W_1 -= update
            elif param_name == 'b_1':
                b_1 -= update
            elif param_name == 'W_2':
                W_2 -= update
            elif param_name == 'b_2':
                b_2 -= update
            elif param_name == 'gamma':
                gamma -= update
            elif param_name == 'beta':
                beta -= update
            elif param_name == 'W_pred':
                W_pred -= update
            elif param_name == 'b_pred':
                b_pred -= update
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {lr:.6f} - Time: {epoch_time:.2f}s")

print("训练完成！")

# ------------------------------------------------------------------
# （5）模型保存
# ------------------------------------------------------------------
print("\n保存模型参数...")
np.savez("./model/transformer_params.npz",
         W_e=W_e, b_e=b_e,
         W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O,
         W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2,
         gamma=gamma, beta=beta,
         W_pred=W_pred, b_pred=b_pred,
         mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)

print("模型已保存到 ./model/transformer_params.npz")