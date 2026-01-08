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
    sample_batches.append(np.array(samples[i:i+B]))
label_batches = []  # (B,)
for i in range(0,len(labels),B):
    label_batches.append(np.array(labels[i:i+B]))

# ------------------------------------------------------------------
# （2）模型参数初始化 - ✅ 所有参数转为 NumPy 数组
# ------------------------------------------------------------------

# 超参数：嵌入维度 d_model
d_model = 128

# 超参数：输入维度 d_in
d_in = 4

# Xavier 初始化
W_e = np.random.randn(d_in,d_model)*np.sqrt(1.0/d_in)  # (4,d_model)
b_e = np.zeros(d_model)                                # (d_model,)

# 位置编码
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

# 注意力头
h = 8
d_K = d_model//h
d_V = d_K

W_Q = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_K = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_V = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_O = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)

# FFN
d_ff = 8 * d_model
W_1 = np.random.randn(d_model,d_ff)*np.sqrt(1.0/d_model)
b_1 = np.zeros(d_ff)
W_2 = np.random.randn(d_ff,d_model)*np.sqrt(1.0/d_ff)
b_2 = np.zeros(d_model)

# 两个独立的 LayerNorm 参数 ✅
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

# 回归头 - ✅ 所有标量转为数组
W_pred = np.random.randn(d_model,1)*np.sqrt(1.0/d_model)
b_pred = np.array([0.0])  # 标量转为 (1,) 数组

# ------------------------------------------------------------------
# （3）辅助函数 - 保持不变
# ------------------------------------------------------------------

def LayerNorm(Z, gamma, beta):
    mean = np.mean(Z,axis=-1,keepdims=True)
    std = np.std(Z,axis=-1,keepdims=True)
    return gamma*((Z-mean)/(std+1e-8))+beta

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

def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    AS = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)
    AS = AS - np.max(AS, axis=-1, keepdims=True)
    exp_AS = np.exp(AS)
    AW = exp_AS / np.sum(exp_AS, axis=-1, keepdims=True)
    out = np.matmul(AW, V_i)
    return out, AW, AS, AW

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
    ASs = []
    V_is = []
    for i in range(h):
        Q_i = Q_iso[:,i,:,:]
        K_i = K_iso[:,i,:,:]
        V_i = V_iso[:,i,:,:]
        out, AW, AS, _ = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)
        AWs.append(AW)
        ASs.append(AS)
        V_is.append(V_i)
    concat_out = np.concatenate(outs, axis=-1)
    outs_MHA = np.matmul(concat_out, W_O)
    return outs_MHA, AWs, ASs, V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V

def Swish(x, beta=1.0):
    sigmoid = 1.0/(1.0+np.exp(-beta*x))
    return x*sigmoid

def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1) + b_1
    A = Swish(L_1)
    L_2 = np.matmul(A, W_2) + b_2
    return L_2, L_1, A

# ------------------------------------------------------------------
# （4）AdamW 优化器类 - ✅ 彻底重构
# ------------------------------------------------------------------

class AdamWOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        初始化 AdamW 优化器
        :param params: 参数字典，key为参数名，value为NumPy数组
        :param lr: 学习率
        :param betas: (beta1, beta2) 动量系数
        :param eps: 数值稳定项
        :param weight_decay: 权重衰减系数
        """
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 状态字典
        self.m = {name: np.zeros_like(param) for name, param in params.items()}
        self.v = {name: np.zeros_like(param) for name, param in params.items()}
        self.t = 0  # 全局时间步
        
        # 权重衰减只应用于权重
        self.weight_params = ['W_e', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2', 'W_pred']
    
    def step(self, grads, lr=None):
        """
        执行单步优化
        :param grads: 梯度字典，与 params 结构相同
        :param lr: 可选的学习率覆盖
        """
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        
        for name in self.params.keys():
            param = self.params[name]
            grad = grads[name]
            
            # 梯度裁剪
            grad = np.clip(grad, -1.0, 1.0)
            
            # 更新一阶矩和二阶矩
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
            
            # 应用更新
            self.params[name] = param - update
    
    def state_dict(self):
        """返回优化器状态，用于保存"""
        return {
            'm': self.m,
            'v': self.v,
            't': self.t
        }
    
    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']

# ------------------------------------------------------------------
# （5）训练循环 - ✅ 使用重构的优化器
# ------------------------------------------------------------------

# 将所有参数放入字典 - ✅ 统一管理
params = {
    'W_e': W_e, 'b_e': b_e,
    'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O,
    'W_1': W_1, 'b_1': b_1, 'W_2': W_2, 'b_2': b_2,
    'gamma1': gamma1, 'beta1': beta1,
    'gamma2': gamma2, 'beta2': beta2,
    'W_pred': W_pred, 'b_pred': b_pred
}

# 初始化 AdamW 优化器
optimizer = AdamWOptimizer(
    params,
    lr=0.005,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

# 超参数
num_epochs = 30
initial_lr = 0.005
final_lr = 1e-6

print("开始训练 (重构版 AdamW 优化器)...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    total_loss = 0.0
    total_samples = 0
    
    # 余弦退火
    if epoch < 5:
        lr = initial_lr
    else:
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * (epoch-5) / (num_epochs-5)))
    
    # 随机打乱批次
    batch_indices = np.random.permutation(len(sample_batches))
    
    for batch_idx in batch_indices:
        X_batch = sample_batches[batch_idx]
        y_true = label_batches[batch_idx]
        B_actual = X_batch.shape[0]
        
        # ------------------------------------------------------------------
        # 前向传播
        # ------------------------------------------------------------------
        
        # 从参数字典获取当前参数
        W_e, b_e = params['W_e'], params['b_e']
        W_Q, W_K, W_V, W_O = params['W_Q'], params['W_K'], params['W_V'], params['W_O']
        W_1, b_1, W_2, b_2 = params['W_1'], params['b_1'], params['W_2'], params['b_2']
        gamma1, beta1 = params['gamma1'], params['beta1']
        gamma2, beta2 = params['gamma2'], params['beta2']
        W_pred, b_pred = params['W_pred'], params['b_pred']
        
        # 线性投影
        E_batch = X_batch @ W_e + b_e
        
        # 注入位置编码
        Z_batch = E_batch + P
        
        # 多头注意力
        outs_MHA, AWs, ASs, V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V = MHA(
            Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        
        # 第一次残差连接 + 层归一化
        res_1 = Z_batch + outs_MHA
        outs_LN_1 = LayerNorm(res_1, gamma1, beta1)
        
        # 前馈网络
        outs_FFN, L_1, A = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
        
        # 第二次残差连接 + 层归一化
        res_2 = outs_LN_1 + outs_FFN
        outs_LN_2 = LayerNorm(res_2, gamma2, beta2)
        
        # 取最后一步 + 回归头
        final_repr = outs_LN_2[:, -1, :]  # (B, d_model)
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)  # (B,)
        
        # 计算损失
        loss = np.mean((y_pred - y_true) ** 2)
        total_loss += loss * B_actual
        total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播
        # ------------------------------------------------------------------
        
        grads = {name: np.zeros_like(param) for name, param in params.items()}
        
        # 1. 回归头
        dL_dy_pred = (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred).reshape(1,)  # 确保是数组
        
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        
        # 2. LayerNorm2 反向传播
        dL_douts_LN2 = np.zeros_like(outs_LN_2)
        dL_douts_LN2[:, -1, :] = dL_dfinal_repr
        _, (dL_dres2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_2, gamma2, beta2, dL_douts_LN2
        )
        grads['gamma2'] = dL_dgamma2
        grads['beta2'] = dL_dbeta2
        
        # 3. 残差 + FFN 反向传播
        dL_douts_LN1 = dL_dres2.copy()
        dL_douts_FFN = dL_dres2.copy()
        
        # FFN 反向传播
        dL_dL2 = dL_douts_FFN
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        
        dL_dA = dL_dL2.reshape(-1, d_model) @ W_2.T
        dL_dA = dL_dA.reshape(B_actual, tau, d_ff)
        
        # Swish 梯度
        sigmoid_L1 = 1.0 / (1.0 + np.exp(-L_1))
        dSwish_dL1 = sigmoid_L1 * (1 + L_1 * (1 - sigmoid_L1))
        dL_dL1 = dL_dA * dSwish_dL1
        
        # FFN 第一层
        grads['W_1'] = outs_LN_1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))
        
        # 正确梯度计算
        dL_douts_LN1_from_FFN = np.matmul(dL_dL1, W_1.T)
        dL_douts_LN1 += dL_douts_LN1_from_FFN
        
        # 4. LayerNorm1 反向传播
        _, (dL_dres1, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            res_1, gamma1, beta1, dL_douts_LN1
        )
        grads['gamma1'] = dL_dgamma1
        grads['beta1'] = dL_dbeta1
        
        # 5. 残差连接1
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
            dL_dout_i = dL_dconcat_out[:, :, i*d_K:(i+1)*d_K]
            AW_i = AWs[i]
            V_i = V_is[i]
            Q_i = Q_iso[:, i, :, :]
            K_i = K_iso[:, i, :, :]
            
            # V 梯度
            dL_dV_i = np.matmul(AW_i.transpose(0,2,1), dL_dout_i)
            
            # AW 梯度
            dL_dAW = np.matmul(dL_dout_i, V_i.transpose(0,2,1))
            
            # AS 梯度
            sum_term = np.sum(dL_dAW * AW_i, axis=-1, keepdims=True)
            dL_dAS = (AW_i * (dL_dAW - sum_term)) / np.sqrt(d_K)
            
            # Q, K 梯度
            dL_dQ_i = np.matmul(dL_dAS, K_i)
            dL_dK_i = np.matmul(dL_dAS.transpose(0,2,1), Q_i)
            
            # 累加
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
        
        # Q, K, V 投影梯度
        grads['W_Q'] = Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # 7. Embedding 层梯度
        dL_dE_batch = dL_dZ_batch
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dE_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dE_batch, axis=(0,1))
        
        # ------------------------------------------------------------------
        # 优化步骤 - ✅ 使用封装的优化器
        # ------------------------------------------------------------------
        optimizer.step(grads, lr=lr)
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {lr:.6f} - Time: {epoch_time:.2f}s")

print("训练完成！")

# ------------------------------------------------------------------
# （6）模型保存
# ------------------------------------------------------------------
print("\n保存模型参数...")
np.savez("./model/transformer_params.npz",
         **params,  # 保存所有参数
         mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)

print("模型已保存到 ./model/transformer_params.npz")