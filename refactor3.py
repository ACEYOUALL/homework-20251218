import numpy as np
import pandas as pd
import time
import os

# 创建模型保存目录
os.makedirs("./model", exist_ok=True)

# ------------------------------------------------------------------
# （1）数据预处理（不变）
# ------------------------------------------------------------------
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values
seq_X = seq[:,:4]
seq_Y = seq[:,4]
mean_X, std_X = seq_X.mean(axis=0), seq_X.std(axis=0)
mean_Y, std_Y = seq_Y.mean(), seq_Y.std()
norm_seq_X = (seq_X-mean_X)/(std_X+1e-8)
norm_seq_Y = (seq_Y-mean_Y)/(std_Y+1e-8)

tau = 10
samples = []
labels = []
for i in range(len(norm_seq_X)-tau):
    samples.append(norm_seq_X[i:i+tau,:])
    labels.append(norm_seq_Y[i+tau])

# 增大批量（从 8→32）
B = 32  # 关键改进：更大的 batch size

sample_batches = []
for i in range(0,len(samples),B):
    sample_batches.append(np.array(samples[i:i+B]))
label_batches = []
for i in range(0,len(labels),B):
    label_batches.append(np.array(labels[i:i+B]))

# ------------------------------------------------------------------
# （2）模型参数初始化（关键修正：独立 LayerNorm 参数）
# ------------------------------------------------------------------

d_model = 128
d_in = 4

W_e = np.random.randn(d_in,d_model)*np.sqrt(1.0/d_in)
b_e = np.zeros(d_model)

# 位置编码（常量）
t = np.arange(tau)[:,np.newaxis]
i = np.arange(0,d_model,2)
div_term = np.exp(i*(-np.log(10000.0)/d_model))
P = np.zeros((tau,d_model))
P[:,0::2] = np.sin(t*div_term)
P[:,1::2] = np.cos(t*div_term)

h = 8
d_K = d_model//h
d_V = d_K

# 为两个 LayerNorm 创建独立参数
gamma1 = np.ones(d_model)  # 用于第一个 LayerNorm (res_1 后)
beta1 = np.zeros(d_model)
gamma2 = np.ones(d_model)  # 用于第二个 LayerNorm (res_2 后)
beta2 = np.zeros(d_model)

# Xavier 初始化其他权重
W_Q = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_K = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_V = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)
W_O = np.random.randn(d_model,d_model)*np.sqrt(1.0/d_model)

d_ff = 8 * d_model
W_1 = np.random.randn(d_model,d_ff)*np.sqrt(1.0/d_model)
b_1 = np.zeros(d_ff)
W_2 = np.random.randn(d_ff,d_model)*np.sqrt(1.0/d_ff)
b_2 = np.zeros(d_model)

W_pred = np.random.randn(d_model,1)*np.sqrt(1.0/d_model)
b_pred = np.zeros(1)

# ------------------------------------------------------------------
# （3）辅助函数（修正 LayerNorm）
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
    AS = np.matmul(Q_i, K_i.transpose(0,2,1)) / np.sqrt(d_K)  # 前向缩放
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
# （4）训练循环（完整修正版）
# ------------------------------------------------------------------

num_epochs = 40  # 增加训练轮数
initial_lr = 0.003  # 增大学习率
final_lr = 1e-5     # 提高最终学习率
weight_decay = 3e-4 # 减小权重衰减
beta1 = 0.9
beta2 = 0.98        # 略微降低 beta2 增加适应性
epsilon = 1e-8
clip_value = 0.5    # 降低裁剪阈值

# AdamW 状态
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
    'gamma1': np.zeros_like(gamma1),  # 独立参数
    'beta1': np.zeros_like(beta1),
    'gamma2': np.zeros_like(gamma2),  # 独立参数
    'beta2': np.zeros_like(beta2),
    'W_pred': np.zeros_like(W_pred),
    'b_pred': np.zeros_like(b_pred)
}

v = {k: np.zeros_like(v) for k,v in m.items()}
t = 0

# 权重参数列表（只对权重应用衰减）
weight_params = ['W_e', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2', 'W_pred']
param_dict = {
    'W_e': W_e, 'W_Q': W_Q, 'W_K': W_K, 'W_V': W_V, 'W_O': W_O,
    'W_1': W_1, 'W_2': W_2, 'W_pred': W_pred
}

print("开始训练 (修正版 AdamW)...")
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    total_loss = 0.0
    total_samples = 0
    
    # 余弦退火（更平滑）
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * epoch / num_epochs))
    
    batch_indices = np.random.permutation(len(sample_batches))
    
    for batch_idx in batch_indices:
        t += 1
        X_batch = sample_batches[batch_idx]
        y_true = label_batches[batch_idx]
        B_actual = X_batch.shape[0]
        
        # ------------------------------------------------------------------
        # 前向传播（使用独立 LayerNorm 参数）
        # ------------------------------------------------------------------
        
        E_batch = X_batch @ W_e + b_e
        Z_batch = E_batch + P  # P 是常量，不参与梯度更新
        
        outs_MHA, AWs, ASs, V_is, Q_iso, K_iso, V_iso, concat_out, Q, K, V = MHA(
            Z_batch, W_Q, W_K, W_V, W_O, h, d_K
        )
        
        # 第一个残差连接 + LayerNorm (使用 gamma1, beta1)
        res_1 = Z_batch + outs_MHA
        outs_LN_1 = LayerNorm(res_1, gamma1, beta1)  # 使用独立参数
        
        outs_FFN, L_1, A = FFN(outs_LN_1, W_1, b_1, W_2, b_2)
        
        # 第二个残差连接 + LayerNorm (使用 gamma2, beta2)
        res_2 = outs_LN_1 + outs_FFN
        outs_LN_2 = LayerNorm(res_2, gamma2, beta2)  # 使用独立参数
        
        final_repr = outs_LN_2[:, -1, :]
        y_pred = (final_repr @ W_pred + b_pred).squeeze(-1)
        
        loss = np.mean((y_pred - y_true) ** 2)
        total_loss += loss * B_actual
        total_samples += B_actual
        
        # ------------------------------------------------------------------
        # 反向传播（完整修正）
        # ------------------------------------------------------------------
        
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
            'gamma1': np.zeros_like(gamma1),  # 独立梯度
            'beta1': np.zeros_like(beta1),
            'gamma2': np.zeros_like(gamma2),  # 独立梯度
            'beta2': np.zeros_like(beta2),
            'W_pred': np.zeros_like(W_pred),
            'b_pred': np.zeros_like(b_pred)
        }
        
        # 1. 回归头
        dL_dy_pred = (y_pred - y_true) / B_actual
        grads['W_pred'] = final_repr.T @ dL_dy_pred.reshape(-1, 1)
        grads['b_pred'] = np.sum(dL_dy_pred)
        dL_dfinal_repr = (dL_dy_pred.reshape(-1, 1) @ W_pred.T).reshape(B_actual, d_model)
        
        # 2. LayerNorm2 反向传播（使用 gamma2, beta2）
        dL_douts_LN2 = np.zeros_like(outs_LN_2)
        dL_douts_LN2[:, -1, :] = dL_dfinal_repr
        _, (dL_dres2, dL_dgamma2, dL_dbeta2) = LayerNorm_with_grad(
            res_2, gamma2, beta2, dL_douts_LN2  # 使用正确的参数
        )
        grads['gamma2'] = dL_dgamma2  # 独立存储
        grads['beta2'] = dL_dbeta2
        
        # 3. 残差 + FFN 反向传播
        dL_douts_LN1 = dL_dres2.copy()
        dL_douts_FFN = dL_dres2.copy()
        
        # FFN 反向传播（修正维度问题）
        dL_dL2 = dL_douts_FFN
        grads['W_2'] = A.reshape(-1, d_ff).T @ dL_dL2.reshape(-1, d_model)
        grads['b_2'] = np.sum(dL_dL2, axis=(0,1))
        
        # 修正：保持三维结构
        dL_dA = np.matmul(dL_dL2, W_2.T).reshape(B_actual, tau, d_ff)  # 直接重塑，不改变维度
        
        sigmoid_L1 = 1.0 / (1.0 + np.exp(-L_1))
        dSwish_dL1 = sigmoid_L1 * (1 + L_1 * (1 - sigmoid_L1))
        dL_dL1 = dL_dA * dSwish_dL1
        
        grads['W_1'] = outs_LN_1.reshape(-1, d_model).T @ dL_dL1.reshape(-1, d_ff)
        grads['b_1'] = np.sum(dL_dL1, axis=(0,1))
        
        # 修正：直接计算梯度，不破坏时间结构
        dL_douts_LN1_from_FFN = np.matmul(dL_dL1, W_1.T)  # (B, tau, d_model)
        dL_douts_LN1 += dL_douts_LN1_from_FFN  # 正确累加
        
        # 4. LayerNorm1 反向传播（使用 gamma1, beta1）
        _, (dL_dres1, dL_dgamma1, dL_dbeta1) = LayerNorm_with_grad(
            res_1, gamma1, beta1, dL_douts_LN1  # 使用正确的参数
        )
        grads['gamma1'] = dL_dgamma1  # 独立存储
        grads['beta1'] = dL_dbeta1
        
        # 5. 残差连接1 反向传播
        dL_dZ_batch = dL_dres1.copy()
        dL_douts_MHA = dL_dres1.copy()
        
        # 6. MHA 反向传播（关键修正：缩放因子）
        grads['W_O'] = concat_out.reshape(-1, d_model).T @ dL_douts_MHA.reshape(-1, d_model)
        dL_dconcat_out = np.matmul(dL_douts_MHA, W_O.T).reshape(B_actual, tau, d_model)
        
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
            
            # AS 梯度（关键：包含缩放因子）
            sum_term = np.sum(dL_dAW * AW_i, axis=-1, keepdims=True)
            dL_dAS = AW_i * (dL_dAW - sum_term) / np.sqrt(d_K)  # 反向也要除以 sqrt(d_K)!
            
            # Q, K 梯度
            dL_dQ_i = np.matmul(dL_dAS, K_i)
            dL_dK_i = np.matmul(dL_dAS.transpose(0,2,1), Q_i)
            
            # 累加到总梯度
            dL_dQ_total[:, :, i*d_K:(i+1)*d_K] += dL_dQ_i
            dL_dK_total[:, :, i*d_K:(i+1)*d_K] += dL_dK_i
            dL_dV_total[:, :, i*d_K:(i+1)*d_K] += dL_dV_i
        
        grads['W_Q'] = Z_batch.reshape(-1, d_model).T @ dL_dQ_total.reshape(-1, d_model)
        grads['W_K'] = Z_batch.reshape(-1, d_model).T @ dL_dK_total.reshape(-1, d_model)
        grads['W_V'] = Z_batch.reshape(-1, d_model).T @ dL_dV_total.reshape(-1, d_model)
        
        # 7. Embedding 层梯度（位置编码 P 是常量，不参与梯度更新）
        dL_dE_batch = dL_dZ_batch  # P 无梯度，直接传递
        grads['W_e'] = X_batch.reshape(-1, d_in).T @ dL_dE_batch.reshape(-1, d_model)
        grads['b_e'] = np.sum(dL_dE_batch, axis=(0,1))
        
        # ------------------------------------------------------------------
        # 优化（AdamW + 梯度检查）
        # ------------------------------------------------------------------
        
        if epoch == 0 and batch_idx == 0:
            print("\n--- 首次迭代梯度检查 ---")
            for name in ['W_pred', 'W_O', 'W_1', 'gamma2']:
                grad_norm = np.linalg.norm(grads[name])
                print(f"Gradient norm for {name}: {grad_norm:.6f}")
                # 预期范围：0.01 ~ 10.0
                if grad_norm < 1e-5:
                    print(f"  ⚠️ 警告：{name} 梯度可能消失！")
                elif grad_norm > 100:
                    print(f"  ⚠️ 警告：{name} 梯度可能爆炸！")
            print("------------------------\n")
        
        # 梯度裁剪
        def safe_clip(grad, min_val, max_val):
            if isinstance(grad, np.ndarray):
                return np.clip(grad, min_val, max_val)
            return np.clip(grad, min_val, max_val)
        
        for param_name, grad in grads.items():
            grads[param_name] = safe_clip(grad, -clip_value, clip_value)
        
        # AdamW 更新
        for param_name in grads.keys():
            g = grads[param_name]
            param_old = eval(param_name)
            
            # 更新动量
            m[param_name] = beta1 * m[param_name] + (1 - beta1) * g
            v[param_name] = beta2 * v[param_name] + (1 - beta2) * (g ** 2)
            
            # 偏差修正
            m_hat = m[param_name] / (1 - beta1 ** t)
            v_hat = v[param_name] / (1 - beta2 ** t)
            
            # AdamW 更新规则
            if param_name in weight_params:
                update = lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * param_old)
            else:
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
            elif param_name == 'gamma1':  # 独立更新
                gamma1 -= update
            elif param_name == 'beta1':
                beta1 -= update
            elif param_name == 'gamma2':  # 独立更新
                gamma2 -= update
            elif param_name == 'beta2':
                beta2 -= update
            elif param_name == 'W_pred':
                W_pred -= update
            elif param_name == 'b_pred':
                b_pred -= update
    
    avg_loss = total_loss / total_samples
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {lr:.6f} - Time: {epoch_time:.2f}s")

print("训练完成！（修正版）")

# ------------------------------------------------------------------
# （5）模型保存
# ------------------------------------------------------------------
print("\n保存模型参数...")
np.savez("./model/transformer_params_fixed.npz",
         W_e=W_e, b_e=b_e,
         W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O,
         W_1=W_1, b_1=b_1, W_2=W_2, b_2=b_2,
         gamma1=gamma1, beta1=beta1,  # 保存独立参数
         gamma2=gamma2, beta2=beta2,
         W_pred=W_pred, b_pred=b_pred,
         mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)

print("修正版模型已保存到 ./model/transformer_params_fixed.npz")