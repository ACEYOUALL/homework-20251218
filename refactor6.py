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
# （2）模型参数初始化 - 三层架构（扁平化参数字典）
# ------------------------------------------------------------------

# 超参数：模型维度
d_model = 64
# 超参数：输入维度
d_in = 4
# 新增：编码器层数
num_layers = 3

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

# 前馈网络维度
d_ff = 4 * d_model

# 【关键修改】为每层初始化独立参数（扁平化字典结构）
params = {
    'W_e': W_e,
    'b_e': b_e,
    'W_pred': KaimingInit((d_model,1), d_model),
    'b_pred': np.array([0.0])
}

# 为每层创建参数
for layer_idx in range(num_layers):
    # MHA参数
    params[f'layer{layer_idx}_W_Q'] = KaimingInit((d_model, d_model), d_model)
    params[f'layer{layer_idx}_W_K'] = KaimingInit((d_model, d_model), d_model)
    params[f'layer{layer_idx}_W_V'] = KaimingInit((d_model, d_model), d_model)
    params[f'layer{layer_idx}_W_O'] = KaimingInit((d_model, d_model), d_model)
    
    # FFN参数
    params[f'layer{layer_idx}_W_1'] = KaimingInit((d_model, d_ff), d_model)
    params[f'layer{layer_idx}_b_1'] = np.zeros(d_ff)
    params[f'layer{layer_idx}_W_2'] = KaimingInit((d_ff, d_model), d_ff)
    params[f'layer{layer_idx}_b_2'] = np.zeros(d_model)
    
    # LayerNorm参数
    params[f'layer{layer_idx}_gamma1'] = np.ones(d_model)
    params[f'layer{layer_idx}_beta1'] = np.zeros(d_model)
    params[f'layer{layer_idx}_gamma2'] = np.ones(d_model)
    params[f'layer{layer_idx}_beta2'] = np.zeros(d_model)

# ------------------------------------------------------------------
# （3）辅助函数（保持不变）
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
        
        # 自动识别权重参数（包含'W_'的参数）
        self.weight_params = [name for name in params.keys() if 'W_' in name or 'W_e' in name or 'W_pred' in name]
    
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
# （5）训练循环 - 三层架构
# ------------------------------------------------------------------

# 优化器
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

print("\n开始训练 (三层Transformer)...")
best_val_loss = float('inf')
grad_print_flag = False  # 每个epoch仅打印一次梯度

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
        
        # ------------------------------------------------------------------
        # 前向传播 (三层)
        # ------------------------------------------------------------------
        # 1. 嵌入层
        E_batch = X_batch @ params['W_e'] + params['b_e']  # (B, τ, d_model)
        Z_batch = E_batch + P  # 添加位置编码
        
        # 保存每层的中间变量
        layer_caches = []
        
        # 2. 通过三层编码器
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
            gamma1 = params[f'layer{layer_idx}_gamma1']
            beta1 = params[f'layer{layer_idx}_beta1']
            gamma2 = params[f'layer{layer_idx}_gamma2']
            beta2 = params[f'layer{layer_idx}_beta2']
            
            # Pre-LN 1: MHA前的LayerNorm
            LN_Z, cache_LN1 = LayerNorm_with_grad(Z_batch, gamma1, beta1)
            
            # MHA
            (outs_MHA, AWs, AS_originals, AS_list, max_AS_list, sum_exp_AS_list,
             V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V) = MHA(
                LN_Z, W_Q, W_K, W_V, W_O, h, d_K
            )
            
            # 残差连接1: 输入 + MHA输出
            res1 = Z_batch + outs_MHA
            
            # Pre-LN 2: FFN前的LayerNorm
            LN_res1, cache_LN2 = LayerNorm_with_grad(res1, gamma2, beta2)
            
            # FFN
            outs_FFN, L_1, A = FFN(LN_res1, W_1, b_1, W_2, b_2)
            
            # 残差连接2: res1 + FFN输出
            Z_batch = res1 + outs_FFN
            
            # 保存中间变量用于反向传播
            layer_caches.append({
                'LN_Z': LN_Z,
                'cache_LN1': cache_LN1,
                'outs_MHA': outs_MHA,
                'AWs': AWs,
                'AS_originals': AS_originals,
                'max_AS_list': max_AS_list,
                'sum_exp_AS_list': sum_exp_AS_list,
                'V_is': V_is,
                'Q_iso': Q_iso,
                'K_iso': K_iso,
                'V_iso': V_iso,
                'concat_out': concat_out,
                'Q': Q,
                'K': K,
                'V': V,
                'res1': res1,
                'LN_res1': LN_res1,
                'cache_LN2': cache_LN2,
                'outs_FFN': outs_FFN,
                'L_1': L_1,
                'A': A,
                'input': Z_batch.copy()  # 保存输入用于梯度检查
            })
        
        # 3. 回归头
        final_repr = np.mean(Z_batch, axis=1)  # (B, d_model)
        y_pred = (final_repr @ params['W_pred'] + params['b_pred']).squeeze(-1)  # (B,)
        
        # 计算MSE损失
        loss = np.mean((y_pred - y_true) ** 2)
        train_total_loss += loss * B_actual
        train_total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播 (三层)
        # ------------------------------------------------------------------
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头梯度
        dL_dy_pred = 2 * (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)
        
        # 平均池化梯度
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ params['W_pred'].T).reshape(B_actual, d_model)
        dL_dlast_layer = np.tile(dL_dfinal_repr[:, np.newaxis, :], (1, tau, 1)) / tau  # (B, τ, d_model)
        
        # 2. 反向传播通过各层
        dL_dZ = dL_dlast_layer
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
            
            # 残差连接2的梯度
            dL_dres1 = dL_dZ.copy()
            dL_douts_FFN = dL_dZ.copy()
            
            # FFN反向
            dL_dL2 = dL_douts_FFN
            grads[f'layer{layer_idx}_W_2'] = cache['A'].reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
            grads[f'layer{layer_idx}_b_2'] = np.sum(dL_dL2, axis=(0,1))
            dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T
            dL_dA = dL_dA.reshape(B_actual, tau, d_ff)
            
            # Swish梯度
            sigmoid_L1 = 1.0 / (1.0 + np.exp(-cache['L_1']))
            dSwish_dL1 = sigmoid_L1 * (1 + cache['L_1'] * (1 - sigmoid_L1))
            dL_dL1 = dL_dA * dSwish_dL1
            
            # FFN第一层梯度
            grads[f'layer{layer_idx}_W_1'] = cache['LN_res1'].reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
            grads[f'layer{layer_idx}_b_1'] = np.sum(dL_dL1, axis=(0,1))
            
            # Pre-LN2 (FFN前)的梯度
            _, (dL_dres1_from_FFN, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
                cache['res1'], gamma2, beta2, dL_dL1 @ W_1.T
            )
            grads[f'layer{layer_idx}_gamma2'] = dL_dgamma2
            grads[f'layer{layer_idx}_beta2'] = dL_dbeta2
            dL_dres1 += dL_dres1_from_FFN
            
            # 残差连接1的梯度
            dL_dinput = dL_dres1.copy()
            dL_douts_MHA = dL_dres1.copy()
            
            # MHA反向
            grads[f'layer{layer_idx}_W_O'] = cache['concat_out'].reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)
            dL_dconcat_out = dL_douts_MHA.reshape(-1, d_model) @ W_O.T
            dL_dconcat_out = dL_dconcat_out.reshape(B_actual, tau, d_model)
            
            dL_dQ_total = np.zeros((B_actual, tau, d_model))
            dL_dK_total = np.zeros((B_actual, tau, d_model))
            dL_dV_total = np.zeros((B_actual, tau, d_model))
            
            for i in range(h):
                dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]
                AW_i = cache['AWs'][i]
                V_i = cache['V_is'][i]
                Q_i = cache['Q_iso'][:, i, :, :]
                K_i = cache['K_iso'][:, i, :, :]
                AS_original_i = cache['AS_originals'][i]
                max_AS_i = cache['max_AS_list'][i]
                sum_exp_AS_i = cache['sum_exp_AS_list'][i]
                
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
            grads[f'layer{layer_idx}_W_Q'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
            grads[f'layer{layer_idx}_W_K'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
            grads[f'layer{layer_idx}_W_V'] = cache['LN_Z'].reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
            
            # Pre-LN1 (MHA前)的梯度
            _, (dL_dinput_from_MHA, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
                cache['input'], gamma1, beta1, 
                dL_dQ_total @ W_Q.T + dL_dK_total @ W_K.T + dL_dV_total @ W_V.T
            )
            grads[f'layer{layer_idx}_gamma1'] = dL_dgamma1
            grads[f'layer{layer_idx}_beta1'] = dL_dbeta1
            dL_dinput += dL_dinput_from_MHA
            
            # 传递到前一层
            dL_dZ = dL_dinput
        
        # 3. 嵌入层梯度
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dZ.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dZ, axis=(0,1))
        
        # ------------------------------------------------------------------
        # 优化器更新
        # ------------------------------------------------------------------
        optimizer.step(grads, lr=lr)
        
        # 梯度调试（每epoch第一次）
        if not grad_print_flag:
            print(f"\n【调试-梯度幅值】Epoch {epoch+1} 第1个批次梯度统计：")
            print(f"  layer0_W_1梯度均值: {grads['layer0_W_1'].mean():.8f}, 绝对值均值: {np.abs(grads['layer0_W_1']).mean():.8f}")
            print(f"  layer2_W_2梯度均值: {grads['layer2_W_2'].mean():.8f}, 绝对值均值: {np.abs(grads['layer2_W_2']).mean():.8f}")
            print(f"  W_pred梯度均值: {grads['W_pred'].mean():.8f}, 绝对值均值: {np.abs(grads['W_pred']).mean():.8f}")
            print(f"\n【调试-中间变量】layer0输出均值: {layer_caches[0]['input'].mean():.6f}, 标准差: {layer_caches[0]['input'].std():.6f}")
            print(f"【调试-中间变量】layer2输出均值: {Z_batch.mean():.6f}, 标准差: {Z_batch.std():.6f}")
            print(f"【调试-中间变量】y_pred均值: {y_pred.mean():.6f}, y_true均值: {y_true.mean():.6f}")
            
            print(f"\n【关键诊断】Layer0输出 STD: {layer_caches[0]['input'].std():.4f}")
            print(f"【关键诊断】Layer2输出 STD: {Z_batch.std():.4f}") 
            
            print(f"Layer0_W_1梯度幅值: {np.abs(grads['layer0_W_1']).mean():.6f}")
            print(f"Layer2_W_1梯度幅值: {np.abs(grads['layer2_W_1']).mean():.6f}")
            
            
            grad_print_flag = True
    
    # 验证阶段
    val_total_loss = 0.0
    val_total_samples = 0
    
    with np.errstate(all='ignore'):
        for batch_idx in range(len(val_sample_batches)):
            X_batch = val_sample_batches[batch_idx]
            y_true = val_label_batches[batch_idx]
            B_actual = X_batch.shape[0]
            
            # 验证前向传播
            E_batch = X_batch @ params['W_e'] + params['b_e']
            Z_batch = E_batch + P
            
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
                
                LN_Z = LayerNorm(Z_batch, gamma1, beta1)
                outs_MHA, _, _, _, _, _, _, _, _, _, _, _, _, _ = MHA(
                    LN_Z, W_Q, W_K, W_V, W_O, h, d_K
                )
                res1 = Z_batch + outs_MHA
                LN_res1 = LayerNorm(res1, gamma2, beta2)
                outs_FFN, _, _ = FFN(LN_res1, W_1, b_1, W_2, b_2)
                Z_batch = res1 + outs_FFN
            
            final_repr = np.mean(Z_batch, axis=1)
            y_pred = (final_repr @ params['W_pred'] + params['b_pred']).squeeze(-1)
            
            loss = np.mean((y_pred - y_true) ** 2)
            val_total_loss += loss * B_actual
            val_total_samples += B_actual
    
    # 结果统计
    avg_train_loss = train_total_loss / train_total_samples
    avg_val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else float('inf')
    epoch_time = time.time() - epoch_start_time
    
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
print("模型已保存到 ./model/，包含三层Transformer参数和归一化统计量")