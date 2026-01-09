import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# （1）数据预处理
# ------------------------------------------------------------------

# 读取原始数据
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

# 分离特征和标签（原始未归一化数据）
seq_X = seq[:,:4]
seq_Y = seq[:,4]

# 超参数：滑动窗口长度 τ
tau = 16

# 构建时序样本和标签（原始数据）
samples = []  # (τ,4)
labels = []   # (scalar)
for i in range(len(seq_X)-tau):
    samples.append(seq_X[i:i+tau,:])
    labels.append(seq_Y[i+tau])

# 划分训练集和验证集（时序数据不随机划分）
split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
train_labels = labels[:split_idx]
val_samples = samples[split_idx:]
val_labels = labels[split_idx:]

# 仅基于训练集计算归一化统计量（解决数据泄露核心问题）
train_samples_np = np.array(train_samples)  # (N_train, τ, 4)
train_labels_np = np.array(train_labels)    # (N_train,)

# 计算训练集特征的均值/标准差（展平为2D计算，保持维度一致）
mean_X_train = train_samples_np.reshape(-1, 4).mean(axis=0)  # (4,)
std_X_train = train_samples_np.reshape(-1, 4).std(axis=0)    # (4,)
# 计算训练集标签的均值/标准差
mean_Y_train = train_labels_np.mean()
std_Y_train = train_labels_np.std()

# 用训练集统计量分别归一化训练集和验证集
# 训练集归一化
norm_train_samples = (train_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_train_labels = (train_labels_np - mean_Y_train) / (std_Y_train + 1e-8)
# 验证集归一化（必须用训练集统计量）
val_samples_np = np.array(val_samples)
val_labels_np = np.array(val_labels)
norm_val_samples = (val_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_val_labels = (val_labels_np - mean_Y_train) / (std_Y_train + 1e-8)

print(f"数据集划分完成 - 训练样本数: {len(norm_train_samples)}, 验证样本数: {len(norm_val_samples)}")
# 打印归一化统计量（调试用）
print(f"【调试-归一化统计】训练集特征均值: {mean_X_train.round(4)}, 标准差: {std_X_train.round(4)}")
print(f"【调试-归一化统计】训练集标签均值: {mean_Y_train:.4f}, 标准差: {std_Y_train:.4f}")
# 【新增调试】打印归一化后的数据分布（确认归一化效果）
print(f"【调试-归一化后分布】训练集特征均值: {norm_train_samples.mean():.6f}, 标准差: {norm_train_samples.std():.6f}")
print(f"【调试-归一化后分布】验证集特征均值: {norm_val_samples.mean():.6f}, 标准差: {norm_val_samples.std():.6f}")
print(f"【调试-归一化后分布】训练集标签均值: {norm_train_labels.mean():.6f}, 验证集标签均值: {norm_val_labels.mean():.6f}")

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
# （2）模型参数初始化
# ------------------------------------------------------------------

# 超参数：模型维度
d_model = 64
# 超参数：输入维度
d_in = 4

# 调整Kaiming初始化增益，适配Swish激活（原ReLU增益不适用）
def KaimingInit(shape, fan_in):
    # Swish激活的最优增益≈sqrt(1/fan_in)，替代原ReLU的sqrt(2/fan_in)
    return np.random.randn(*shape) * np.sqrt(1.0 / fan_in)

# 特征嵌入层线性投影的权重和偏置
W_e = KaimingInit((d_in, d_model), d_in)  # (4,d_model)
b_e = np.zeros(d_model)                   # (d_model,)

# 位置编码（保持不变）
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))  # (τ,d_model)
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

# 注意力头数
h = 16
# 单头维度
d_K = d_model//h
d_V = d_K

# 初始化注意力层参数
W_Q = KaimingInit((d_model, d_model), d_model)
W_K = KaimingInit((d_model, d_model), d_model)
W_V = KaimingInit((d_model, d_model), d_model)
W_O = KaimingInit((d_model, d_model), d_model)

# 前馈网络维度
d_ff = 4 * d_model
# 初始化FNN参数
W_1 = KaimingInit((d_model, d_ff), d_model)
b_1 = np.zeros(d_ff)
W_2 = KaimingInit((d_ff, d_model), d_ff)
b_2 = np.zeros(d_model)

# LayerNorm参数
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

# 回归头参数
W_pred = KaimingInit((d_model,1), d_model)
b_pred = np.array([0.0])

# 【新增调试】打印初始参数统计（确认初始化效果）
print(f"\n【调试-初始参数】W_1均值: {W_1.mean():.6f}, 标准差: {W_1.std():.6f}")
print(f"【调试-初始参数】W_2均值: {W_2.mean():.6f}, 标准差: {W_2.std():.6f}")
print(f"【调试-初始参数】W_pred均值: {W_pred.mean():.6f}, 标准差: {W_pred.std():.6f}")

# ------------------------------------------------------------------
# （3）辅助函数
# ------------------------------------------------------------------

# 层归一化（核心函数保持不变）
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

# 缩放点积注意力（保持不变）
def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    AS_original = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)
    max_AS = np.max(AS_original, axis=-1, keepdims=True)
    AS = AS_original - max_AS
    exp_AS = np.exp(AS)
    sum_exp_AS = np.sum(exp_AS, axis=-1, keepdims=True)
    AW = exp_AS / (sum_exp_AS + 1e-8)
    out = np.matmul(AW, V_i)
    return out, AW, AS_original, AS, max_AS, sum_exp_AS

# 多头注意力实现（保持不变）
def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    B, tau, _ = Z.shape
    Q = np.matmul(Z, W_Q)
    K = np.matmul(Z, W_K)
    V = np.matmul(Z, W_V)
    
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0,2,1,3)
    
    outs = []
    AWs = []
    AS_originals = []
    AS_list = []
    max_AS_list = []
    sum_exp_AS_list = []
    V_is = []
    
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
    
    concat_out = np.concatenate(outs, axis=-1)
    outs_MHA = np.matmul(concat_out, W_O)
    return (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list, 
            V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V)

# Swish激活函数（保持不变）
def Swish(x, beta=1.0):
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x*sigmoid

# 前馈网络实现（保持不变）
def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1) + b_1
    A = Swish(L_1)
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

# 调整初始学习率（适配梯度幅值提升）
optimizer = AdamWOptimizer(
    params,
    lr=1e-3,  # 从5e-4提升到1e-3，让小梯度也能产生有效更新
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

# 训练超参数
num_epochs = 100   
initial_lr = 1e-3  
final_lr = 5e-4    

print("\n开始训练...")
best_val_loss = float('inf')
# 【新增】标记是否打印过梯度（每个Epoch仅打印一次，避免刷屏）
grad_print_flag = False

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_total_loss = 0.0
    train_total_samples = 0
    
    # 余弦退火学习率
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / num_epochs))
    
    # 训练阶段
    train_batch_indices = np.random.permutation(len(train_sample_batches))
    
    for batch_idx in train_batch_indices:
        X_batch = train_sample_batches[batch_idx]
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
        
        # Pre-LN：先归一化再做MHA（Transformer标准做法，稳定梯度）
        LN_Z_batch = LayerNorm(Z_batch, gamma1, beta1)
        (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list,
         V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V) = MHA(
            LN_Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        # 残差连接（Pre-LN：输入直接残差，而非归一化后）
        res_1 = Z_batch + outs_MHA
        
        # Pre-LN：FFN层先归一化再计算
        LN_res1 = LayerNorm(res_1, gamma2, beta2)
        outs_FFN, L_1, A = FFN(LN_res1, W_1, b_1, W_2, b_2)
        # 第二次残差连接
        res_2 = res_1 + outs_FFN
        
        # 回归头：平均池化替代仅取最后时间步（解决梯度稀释）
        final_repr = np.mean(res_2, axis=1)  # (B, d_model)，时序维度平均
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
        
        # 计算MSE损失
        loss = np.mean((y_pred - y_true) ** 2)
        train_total_loss += loss * B_actual
        train_total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播
        # ------------------------------------------------------------------
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头梯度（适配平均池化）
        dL_dy_pred = 2 * (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)
        # 平均池化的梯度：均匀分配到所有时间步
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        dL_dres2 = np.tile(dL_dfinal_repr[:, np.newaxis, :], (1, tau, 1)) / tau  # (B, τ, d_model)
        
        # 2. 第二次残差连接 + FFN反向传播（适配Pre-LN）
        dL_dres1 = dL_dres2.copy()
        dL_douts_FFN = dL_dres2.copy()
        
        # FFN反向
        dL_dL2 = dL_douts_FFN
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T
        dL_dA = dL_dA.reshape(B_actual, tau, d_ff)
        
        # Swish梯度
        sigmoid_L1 = 1.0 / (1.0 + np.exp(-L_1))
        dSwish_dL1 = sigmoid_L1 * (1 + L_1 * (1 - sigmoid_L1))
        dL_dL1 = dL_dA * dSwish_dL1
        
        # FFN第一层梯度
        grads['W_1'] = LN_res1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))
        
        # Pre-LN：LN_res1的梯度
        _, (dL_dres1_from_FFN, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_1, gamma2, beta2, dL_dL1 @ W_1.T
        )
        grads['gamma2'] = dL_dgamma2
        grads['beta2'] = dL_dbeta2
        dL_dres1 += dL_dres1_from_FFN
        
        # 3. 第一次残差连接 + MHA反向传播（适配Pre-LN）
        dL_dZ_batch = dL_dres1.copy()
        dL_douts_MHA = dL_dres1.copy()
        
        # MHA反向
        grads['W_O'] = concat_out.reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)
        dL_dconcat_out = dL_douts_MHA.reshape(-1, d_model) @ W_O.T
        dL_dconcat_out = dL_dconcat_out.reshape(B_actual, tau, d_model)
        
        dL_dQ_total = np.zeros((B_actual, tau, d_model))
        dL_dK_total = np.zeros((B_actual, tau, d_model))
        dL_dV_total = np.zeros((B_actual, tau, d_model))
        
        for i in range(h):
            dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]
            AW_i = AWs[i]
            V_i = V_is[i]
            Q_i = Q_iso[:, i, :, :]
            K_i = K_iso[:, i, :, :]
            AS_original_i = AS_originals[i]
            AS_i = AS_list[i]
            max_AS_i = max_AS_list[i]
            sum_exp_AS_i = sum_exp_AS_list[i]
            
            dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)
            dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))
            dL_dAS = AW_i * (dL_dAW - np.sum(dL_dAW * AW_i, axis=-1, keepdims=True))
            dL_dmax_AS = np.sum(dL_dAS, axis=-1, keepdims=True)
            mask = (AS_original_i == max_AS_i).astype(np.float32)
            mask_sum = np.sum(mask, axis=-1, keepdims=True) + 1e-8
            dL_dAS_original = dL_dAS - mask * dL_dmax_AS / mask_sum
            
            dL_dQ_i = np.matmul(dL_dAS_original / np.sqrt(d_K), K_i)
            dL_dK_i = np.matmul(dL_dAS_original.transpose(0,2,1) / np.sqrt(d_K), Q_i)
            
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
        
        # Q/K/V投影梯度
        grads['W_Q'] = LN_Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = LN_Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = LN_Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # Pre-LN：LN_Z_batch的梯度
        _, (dL_dZ_batch_from_MHA, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            Z_batch, gamma1, beta1, dL_dQ_total @ W_Q.T + dL_dK_total @ W_K.T + dL_dV_total @ W_V.T
        )
        grads['gamma1'] = dL_dgamma1
        grads['beta1'] = dL_dbeta1
        dL_dZ_batch += dL_dZ_batch_from_MHA
        
        # 4. Embedding层梯度
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dZ_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dZ_batch, axis=(0,1))
        
        # ------------------------------------------------------------------
        # 优化器更新
        # ------------------------------------------------------------------
        optimizer.step(grads, lr=lr)
        
        # 【新增调试】每个Epoch仅打印一次梯度统计（避免刷屏）
        if not grad_print_flag:
            print(f"\n【调试-梯度幅值】Epoch {epoch+1} 第1个批次梯度统计：")
            print(f"  W_1梯度均值: {grads['W_1'].mean():.8f}, 绝对值均值: {np.abs(grads['W_1']).mean():.8f}")
            print(f"  W_2梯度均值: {grads['W_2'].mean():.8f}, 绝对值均值: {np.abs(grads['W_2']).mean():.8f}")
            print(f"  W_pred梯度均值: {grads['W_pred'].mean():.8f}, 绝对值均值: {np.abs(grads['W_pred']).mean():.8f}")
            print(f"  W_e梯度均值: {grads['W_e'].mean():.8f}, W_Q梯度均值: {grads['W_Q'].mean():.8f}")
            # 打印中间变量分布（确认Pre-LN稳定性）
            print(f"\n【调试-中间变量】LN_Z_batch均值: {LN_Z_batch.mean():.6f}, 标准差: {LN_Z_batch.std():.6f}")
            print(f"【调试-中间变量】res_2均值: {res_2.mean():.6f}, 标准差: {res_2.std():.6f}")
            print(f"【调试-中间变量】y_pred均值: {y_pred.mean():.6f}, y_true均值: {y_true.mean():.6f}")
            grad_print_flag = True
    
    # 验证阶段
    val_total_loss = 0.0
    val_total_samples = 0
    
    with np.errstate(all='ignore'):
        for batch_idx in range(len(val_sample_batches)):
            X_batch = val_sample_batches[batch_idx]
            y_true = val_label_batches[batch_idx]
            B_actual = X_batch.shape[0]
            
            # 验证阶段前向传播（适配Pre-LN和平均池化）
            W_e, b_e = params['W_e'], params['b_e']
            W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
            W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
            gamma1, beta1 = params['gamma1'], params['beta1']
            gamma2, beta2 = params['gamma2'], params['beta2']
            W_pred, b_pred = params['W_pred'], params['b_pred']
            
            E_batch = X_batch @ W_e + b_e
            Z_batch = E_batch + P
            
            # Pre-LN + MHA
            LN_Z_batch = LayerNorm(Z_batch, gamma1, beta1)
            outs_MHA, _, _, _, _, _, _, _, _, _, _, _, _, _ = MHA(
                LN_Z_batch, W_Q, W_K, W_V, W_O, h, d_K
            )
            res_1 = Z_batch + outs_MHA
            
            # Pre-LN + FFN
            LN_res1 = LayerNorm(res_1, gamma2, beta2)
            outs_FFN, _, _ = FFN(LN_res1, W_1, b_1, W_2, b_2)
            res_2 = res_1 + outs_FFN
            
            # 平均池化回归头
            final_repr = np.mean(res_2, axis=1)
            y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
            
            loss = np.mean((y_pred - y_true) ** 2)
            val_total_loss += loss * B_actual
            val_total_samples += B_actual
    
    # 结果统计
    avg_train_loss = train_total_loss / train_total_samples
    avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else float('inf')
    epoch_time = time.time() - epoch_start_time
    
    # 【新增调试】每个Epoch打印参数更新后的统计
    print(f"\n【调试-Epoch{epoch+1}参数】W_1均值: {params['W_1'].mean():.6f}, W_2均值: {params['W_2'].mean():.6f}")
    print(f"【调试-Epoch{epoch+1}参数】W_pred均值: {params['W_pred'].mean():.6f}, b_pred: {params['b_pred'][0]:.6f}")
    
    # 保存最优模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # 保存时包含训练集的归一化统计量
        np.savez("./model/best_transformer_params.npz",
                 **params, 
                 mean_X_train=mean_X_train, std_X_train=std_X_train,
                 mean_Y_train=mean_Y_train, std_Y_train=std_Y_train)
        print(f"✅ 最优验证损失更新: {best_val_loss:.6f}，已保存模型")
    
    # 打印训练信息
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.6f} - "
          f"Val Loss: {avg_val_loss:.6f} - "
          f"Best Val Loss: {best_val_loss:.6f} - "
          f"LR: {lr:.6f} - Time: {epoch_time:.2f}s")
    
    # 重置梯度打印标记（下一个Epoch重新打印）
    grad_print_flag = False

print("\n训练完成！")
print(f"最优验证损失: {best_val_loss:.6f}")

# 保存最终模型（包含训练集统计量）
np.savez("./model/transformer_params.npz",
         **params,
         mean_X_train=mean_X_train, std_X_train=std_X_train,
         mean_Y_train=mean_Y_train, std_Y_train=std_Y_train)
print("模型已保存到 ./model/，包含训练集归一化统计量")