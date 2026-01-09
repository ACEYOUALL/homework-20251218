import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# （1）数据预处理
# ------------------------------------------------------------------

seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 分离标签
seq_X = seq[:,:4]
seq_Y = seq[:,4]

# Z-score 归一化
mean_X, std_X = seq_X.mean(axis=0), seq_X.std(axis=0)
mean_Y, std_Y = seq_Y.mean(), seq_Y.std()
norm_seq_X = (seq_X - mean_X) / (std_X + 1e-8)
norm_seq_Y = (seq_Y - mean_Y) / (std_Y + 1e-8)

# 超参数：滑动窗口长度 τ
tau = 16

# 样本和标签
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(norm_seq_X)-tau):
    samples.append(norm_seq_X[i:i+tau,:])
    labels.append(norm_seq_Y[i+tau])

# ✅ 新增：划分训练集和验证集（80%训练，20%验证，时序数据不随机划分）
split_idx = int(len(samples) * 0.8)  # 按8:2划分
train_samples = samples[:split_idx]
train_labels = labels[:split_idx]
val_samples = samples[split_idx:]
val_labels = labels[split_idx:]

print(f"数据集划分完成 - 训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")  # ✅ 新增

# 超参数：批量 B
B = 32

# ✅ 修改：生成训练集批次
train_sample_batches = []  # (B,τ,4)
for i in range(0, len(train_samples), B):
    train_sample_batches.append(np.array(train_samples[i:i+B]))
train_label_batches = []  # (B,)
for i in range(0, len(train_labels), B):
    train_label_batches.append(np.array(train_labels[i:i+B]))

# ✅ 新增：生成验证集批次
val_sample_batches = []  # (B,τ,4)
for i in range(0, len(val_samples), B):
    val_sample_batches.append(np.array(val_samples[i:i+B]))
val_label_batches = []  # (B,)
for i in range(0, len(val_labels), B):
    val_label_batches.append(np.array(val_labels[i:i+B]))

# ------------------------------------------------------------------
# （2）模型参数初始化
# ------------------------------------------------------------------

# 超参数：模型维度
d_model = 16

# 超参数：输入维度
d_in = 4

# Kaiming 初始化方法
def KaimingInit(shape, fan_in):
    return np.random.randn(*shape) * np.sqrt(2.0 / fan_in)

# 特征嵌入层线性投影的权重和偏置
W_e = KaimingInit((d_in, d_model), d_in)  # (4,d_model)
b_e = np.zeros(d_model)                   # (d_model,)

# 位置编码
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))  # (τ,d_model)
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

# 注意力头数
h = 4

# 单头维度
d_K = d_model//h
d_V = d_K

# Kaiming 初始化注意力层，(d_model,d_model)
W_Q = KaimingInit((d_model, d_model), d_model)
W_K = KaimingInit((d_model, d_model), d_model)
W_V = KaimingInit((d_model, d_model), d_model)
W_O = KaimingInit((d_model, d_model), d_model)

# 前馈维度
d_ff = 4 * d_model  # 8 倍 d_model

# 初始化 FNN 两次线性投影的权重和偏置
W_1 = KaimingInit((d_model, d_ff), d_model)
b_1 = np.zeros(d_ff)
W_2 = KaimingInit((d_ff, d_model), d_ff)
b_2 = np.zeros(d_model)

# LayerNorm 参数
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

# Kaiming 初始化回归头
W_pred = KaimingInit((d_model,1), d_model)
b_pred = np.array([0.0])

# ------------------------------------------------------------------
# （3）辅助函数
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

# 缩放点积注意力
def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    # 注意这里 Q_i、K_i、V_i 是单头，(B,τ,d_K)
    # 注意力得分
    AS_original = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)
    # 利用 Softmax 的平移性质避免 exp 溢出
    max_AS = np.max(AS_original, axis=-1, keepdims=True)
    AS = AS_original - max_AS
    # Softmax 计算注意力权重
    exp_AS = np.exp(AS)
    sum_exp_AS = np.sum(exp_AS, axis=-1, keepdims=True)
    AW = exp_AS / (sum_exp_AS + 1e-8)
    # 单头注意力输出，(B,τ,d_V)
    out = np.matmul(AW, V_i)
    # 返回所有中间变量用于反向传播
    return out, AW, AS_original, AS, max_AS, sum_exp_AS

# 多头注意力实现
def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    B, tau, _ = Z.shape
    # 计算 Q、K、V，(B,τ,d_model)
    Q = np.matmul(Z, W_Q)
    K = np.matmul(Z, W_K)
    V = np.matmul(Z, W_V)
    # 分离 Q、K、V
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    # 注意力结果，(B,τ,d_V)
    outs = []
    # 注意力权重，(B,τ,τ)
    AWs = []
    AS_originals = []
    AS_list = []
    max_AS_list = []
    sum_exp_AS_list = []
    V_is = []
    # 计算单头注意力
    for i in range(h):
        Q_i = Q_iso[:,i,:,:]
        K_i = K_iso[:,i,:,:]
        V_i = V_iso[:,i,:,:]
        out, AW, AS_original, AS, max_AS, sum_exp_AS = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)
        AWs.append(AW)
        AS_originals.append(AS_original)
        AS_list.append(AS)
        max_AS_list.append(max_AS)
        sum_exp_AS_list.append(sum_exp_AS)
        V_is.append(V_i)
    # 拼接并得到多头结果，(B,τ,d_V·h) → (B,τ,d_model)
    concat_out = np.concatenate(outs, axis=-1)
    outs_MHA = np.matmul(concat_out, W_O)
    # 返回所有中间变量用于反向传播
    return (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list, 
            V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V)

# Swish 激活函数
def Swish(x, beta=1.0):
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x*sigmoid

# 前馈网络实现
def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1) + b_1
    A = Swish(L_1)
    L_2 = np.matmul(A, W_2) + b_2
    return L_2, L_1, A

# ------------------------------------------------------------------
# （4）AdamW 优化器
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
            
            # 梯度裁剪
            grad = np.clip(grad, -0.1, 0.1)
            
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
# （5）训练循环
# ------------------------------------------------------------------

# 参数字典
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

# 实例化一个 AdamW 优化器
optimizer = AdamWOptimizer(
    params,
    lr=5e-4,  # 学习率
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4 # AdamW 的权重衰减
)

# 训练超参数
num_epochs = 100   # 训练轮数
initial_lr = 5e-4  # 初始学习率
final_lr = 1e-5    # 终末学习率

print("开始训练...")

# ✅ 修改：最优损失改为验证集损失
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 记录训练信息
    epoch_start_time = time.time()
    train_total_loss = 0.0  # ✅ 修改：区分训练损失
    train_total_samples = 0
    
    # 以余弦退火调度学习率
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / num_epochs))
    
    # ---------------------- 训练阶段 ----------------------
    # 随机打乱训练批次（验证集不打乱）
    train_batch_indices = np.random.permutation(len(train_sample_batches))
    
    for batch_idx in train_batch_indices:
        X_batch = train_sample_batches[batch_idx]  # ✅ 修改：用训练集批次
        y_true = train_label_batches[batch_idx]
        B_actual = X_batch.shape[0]
        
        # 从字典获取当前参数
        W_e, b_e = params['W_e'], params['b_e']
        W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
        W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
        gamma1, beta1 = params['gamma1'], params['beta1']
        gamma2, beta2 = params['gamma2'], params['beta2']
        W_pred, b_pred = params['W_pred'], params['b_pred']
        
        # ------------------------------------------------------------------
        # 前向传播
        # ------------------------------------------------------------------
        E_batch = X_batch @ W_e + b_e
        Z_batch = E_batch + P
        
        # 计算多头注意力
        (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list,
         V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V) = MHA(
            Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        
        # 第一次残差连接、层归一化
        res_1 = Z_batch + outs_MHA
        outs_LN_1 = LayerNorm(res_1, gamma1, beta1)
        
        # FFN
        outs_FFN, L_1, A = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
        
        # 第二次残差连接、层归一化
        res_2 = outs_LN_1 + outs_FFN
        outs_LN_2 = LayerNorm(res_2, gamma2, beta2)
        
        # 回归头
        final_repr = outs_LN_2[:, -1, :]
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
        
        # 计算 MSE 损失
        loss = np.mean((y_pred - y_true) ** 2)
        train_total_loss += loss * B_actual
        train_total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播
        # ------------------------------------------------------------------
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头梯度
        dL_dy_pred = 2 * (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        
        # 2. LayerNorm2 反向传播
        dL_douts_LN2 = np.zeros_like(outs_LN_2)
        dL_douts_LN2[:, -1, :] = dL_dfinal_repr
        _, (dL_dres2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_2, gamma2, beta2, dL_douts_LN2
        )
        grads['gamma2'] = dL_dgamma2
        grads['beta2'] = dL_dbeta2
        
        # 3. FFN 以及残差连接的反向传播
        dL_douts_LN1 = dL_dres2.copy()
        dL_douts_FFN = dL_dres2.copy()
        dL_dL2 = dL_douts_FFN
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T
        dL_dA = dL_dA.reshape(B_actual, tau, d_ff)
        
        # Swish 梯度
        sigmoid_L1 = 1.0 / (1.0 + np.exp(-L_1))
        dSwish_dL1 = sigmoid_L1 * (1 + L_1 * (1 - sigmoid_L1))
        dL_dL1 = dL_dA * dSwish_dL1
        
        # FFN 第一层梯度
        grads['W_1'] = outs_LN_1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))
        
        # 累加 LN1 梯度
        dL_douts_LN1_from_FFN = np.matmul(dL_dL1, W_1.T)
        dL_douts_LN1 += dL_douts_LN1_from_FFN
        
        # 4. LayerNorm1 反向传播
        _, (dL_dres1, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            res_1, gamma1, beta1, dL_douts_LN1
        )
        grads['gamma1'] = dL_dgamma1
        grads['beta1'] = dL_dbeta1
        
        # 5. 残差连接 1
        dL_dZ_batch = dL_dres1.copy()
        dL_douts_MHA = dL_dres1.copy()
        
        # 6. MHA 反向传播
        grads['W_O'] = concat_out.reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)
        dL_dconcat_out = dL_douts_MHA.reshape(-1, d_model) @ W_O.T
        dL_dconcat_out = dL_dconcat_out.reshape(B_actual, tau, d_model)
        
        dL_dQ_total = np.zeros((B_actual, tau, d_model))
        dL_dK_total = np.zeros((B_actual, tau, d_model))
        dL_dV_total = np.zeros((B_actual, tau, d_model))
        
        for i in range(h):
            # 取出当前头的梯度和中间变量
            dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]
            AW_i = AWs[i]
            V_i = V_is[i]
            Q_i = Q_iso[:, i, :, :]
            K_i = K_iso[:, i, :, :]
            AS_original_i = AS_originals[i]
            AS_i = AS_list[i]
            max_AS_i = max_AS_list[i]
            sum_exp_AS_i = sum_exp_AS_list[i]
            
            # 计算 dL_dV_i
            dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)
            
            # 计算 dL_dAW_i
            dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))
            
            # 计算 dL_dAS_i
            dL_dAS = AW_i * (dL_dAW - np.sum(dL_dAW * AW_i, axis=-1, keepdims=True))
            
            # 计算 dL_dAS_original_i
            dL_dmax_AS = np.sum(dL_dAS, axis=-1, keepdims=True)
            mask = (AS_original_i == max_AS_i).astype(np.float32)
            mask_sum = np.sum(mask, axis=-1, keepdims=True) + 1e-8
            dL_dAS_original = dL_dAS - mask * dL_dmax_AS / mask_sum
            
            # 计算 Q/K 梯度
            dL_dQ_i = np.matmul(dL_dAS_original / np.sqrt(d_K), K_i)
            dL_dK_i = np.matmul(dL_dAS_original.transpose(0,2,1) / np.sqrt(d_K), Q_i)
            
            # 累加梯度
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
        
        # Q/K/V 投影梯度
        grads['W_Q'] = Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # 7. Embedding层梯度
        dL_dE_batch = dL_dZ_batch
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dE_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dE_batch, axis=(0,1))
        
        # ------------------------------------------------------------------
        # AdamW 优化
        # ------------------------------------------------------------------
        optimizer.step(grads, lr=lr)
    
    # ---------------------- 验证阶段 ----------------------
    # ✅ 新增：验证集前向传播（无反向传播，无参数更新）
    val_total_loss = 0.0
    val_total_samples = 0
    
    # 验证阶段禁用梯度（这里通过不计算梯度实现）
    with np.errstate(all='ignore'):  # 抑制数值警告
        for batch_idx in range(len(val_sample_batches)):
            X_batch = val_sample_batches[batch_idx]
            y_true = val_label_batches[batch_idx]
            B_actual = X_batch.shape[0]
            
            # 前向传播（仅计算损失，不反向传播）
            W_e, b_e = params['W_e'], params['b_e']
            W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
            W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
            gamma1, beta1 = params['gamma1'], params['beta1']
            gamma2, beta2 = params['gamma2'], params['beta2']
            W_pred, b_pred = params['W_pred'], params['b_pred']
            
            E_batch = X_batch @ W_e + b_e
            Z_batch = E_batch + P
            
            # 仅前向计算MHA（不需要中间变量）
            outs_MHA, _, _, _, _, _, _, _, _, _, _, _, _, _ = MHA(
                Z_batch, W_Q, W_K, W_V, W_O, h, d_K
            )
            
            res_1 = Z_batch + outs_MHA
            outs_LN_1 = LayerNorm(res_1, gamma1, beta1)
            
            outs_FFN, _, _ = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
            
            res_2 = outs_LN_1 + outs_FFN
            outs_LN_2 = LayerNorm(res_2, gamma2, beta2)
            
            final_repr = outs_LN_2[:, -1, :]
            y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
            
            # 计算验证损失
            loss = np.mean((y_pred - y_true) ** 2)
            val_total_loss += loss * B_actual
            val_total_samples += B_actual
    
    # ---------------------- 结果统计 ----------------------
    # 计算训练和验证平均损失
    avg_train_loss = train_total_loss / train_total_samples
    avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else float('inf')
    epoch_time = time.time() - epoch_start_time
    
    # ✅ 修改：基于验证损失保存最优模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        np.savez("./model/best_transformer_params.npz",
                 **params, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)
        print(f"✅ 最优验证损失更新: {best_val_loss:.6f}，已保存模型")
    
    # 打印训练信息（包含训练/验证损失）
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.6f} - "
          f"Val Loss: {avg_val_loss:.6f} - "
          f"Best Val Loss: {best_val_loss:.6f} - "
          f"LR: {lr:.6f} - Time: {epoch_time:.2f}s")

print("训练完成！")
print(f"最优验证损失: {best_val_loss:.6f}")

# 保存最终模型
np.savez("./model/transformer_params.npz",
         **params, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)
print("模型已保存到 ./model/")