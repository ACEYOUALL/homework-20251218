import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. 配置超参数（与训练代码完全一致）
# ------------------------------------------------------------------
tau = 16          # 滑动窗口长度
d_model = 64      # 模型维度
d_in = 4          # 输入特征维度
h = 16            # 注意力头数
d_K = d_model // h  # 单头维度
d_V = d_K
d_ff = 4 * d_model  # FFN隐藏层维度

# ------------------------------------------------------------------
# 2. 复制训练代码中的核心辅助函数（保证前向逻辑一致）
# ------------------------------------------------------------------
def LayerNorm(Z, gamma, beta):
    mean = np.mean(Z, axis=-1, keepdims=True)
    std = np.std(Z, axis=-1, keepdims=True)
    return gamma * ((Z - mean) / (std + 1e-8)) + beta

def ScaledDotProductAttention(Q_i, K_i, V_i, d_K):
    AS_original = np.matmul(Q_i, K_i.transpose(0, 2, 1)) / np.sqrt(d_K)
    max_AS = np.max(AS_original, axis=-1, keepdims=True)
    AS = AS_original - max_AS
    exp_AS = np.exp(AS)
    sum_exp_AS = np.sum(exp_AS, axis=-1, keepdims=True)
    AW = exp_AS / (sum_exp_AS + 1e-8)
    out = np.matmul(AW, V_i)
    return out

def MHA(Z, W_Q, W_K, W_V, W_O, h, d_K):
    B, tau, _ = Z.shape
    Q = np.matmul(Z, W_Q)
    K = np.matmul(Z, W_K)
    V = np.matmul(Z, W_V)
    
    Q_iso = Q.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    K_iso = K.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    V_iso = V.reshape(B, tau, h, d_K).transpose(0, 2, 1, 3)
    
    outs = []
    for i in range(h):
        Q_i = Q_iso[:, i, :, :]
        K_i = K_iso[:, i, :, :]
        V_i = V_iso[:, i, :, :]
        out = ScaledDotProductAttention(Q_i, K_i, V_i, d_K)
        outs.append(out)
    
    concat_out = np.concatenate(outs, axis=-1)
    outs_MHA = np.matmul(concat_out, W_O)
    return outs_MHA

def Swish(x, beta=1.0):
    sigmoid = 1.0 / (1.0 + np.exp(-beta * x))
    return x * sigmoid

def FFN(Z, W_1, b_1, W_2, b_2):
    L_1 = np.matmul(Z, W_1) + b_1
    A = Swish(L_1)
    L_2 = np.matmul(A, W_2) + b_2
    return L_2

# ------------------------------------------------------------------
# 3. 加载模型参数和归一化统计量
# ------------------------------------------------------------------
print("开始加载模型参数...")
model_path = "./model/transformer_params.npz"
# 加载npz文件
params_dict = np.load(model_path, allow_pickle=True)

# 提取模型权重
W_e = params_dict['W_e']
b_e = params_dict['b_e']
W_Q = params_dict['W_Q']
W_K = params_dict['W_K']
W_V = params_dict['W_V']
W_O = params_dict['W_O']
W_1 = params_dict['W_1']
b_1 = params_dict['b_1']
W_2 = params_dict['W_2']
b_2 = params_dict['b_2']
gamma1 = params_dict['gamma1']
beta1 = params_dict['beta1']
gamma2 = params_dict['gamma2']
beta2 = params_dict['beta2']
W_pred = params_dict['W_pred']
b_pred = params_dict['b_pred']

# 提取训练集归一化统计量（关键！必须用训练集的均值/标准差）
mean_X_train = params_dict['mean_X_train']
std_X_train = params_dict['std_X_train']
mean_Y_train = params_dict['mean_Y_train']
std_Y_train = params_dict['std_Y_train']

print("模型参数加载完成！")
print(f"训练集标签均值: {mean_Y_train:.4f}, 标准差: {std_Y_train:.4f}")

# ------------------------------------------------------------------
# 4. 数据预处理：构建训练集和验证集（与训练代码完全一致）
# ------------------------------------------------------------------
# 读取原始数据
print("\n读取原始数据并构建训练集/验证集...")
seq = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values
seq_X = seq[:, :4]  # 特征
seq_Y = seq[:, 4]   # 标签

# 构建完整的时序样本（和训练代码一致）
samples = []
labels = []
for i in range(len(seq_X) - tau):
    samples.append(seq_X[i:i+tau, :])
    labels.append(seq_Y[i+tau])

# 划分训练集和验证集（前80%训练，后20%验证）
split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
train_labels = labels[:split_idx]
val_samples = samples[split_idx:]
val_labels = labels[split_idx:]

# 转换为numpy数组
train_samples_np = np.array(train_samples)  # (N_train, tau, 4)
train_labels_np = np.array(train_labels)    # (N_train,)
val_samples_np = np.array(val_samples)      # (N_val, tau, 4)
val_labels_np = np.array(val_labels)        # (N_val,)

# 用训练集统计量归一化
# 训练集归一化
norm_train_samples = (train_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_train_labels = (train_labels_np - mean_Y_train) / (std_Y_train + 1e-8)
# 验证集归一化
norm_val_samples = (val_samples_np - mean_X_train) / (std_X_train + 1e-8)
norm_val_labels = (val_labels_np - mean_Y_train) / (std_Y_train + 1e-8)

print(f"训练集构建完成 - 样本数: {len(norm_train_samples)}")
print(f"验证集构建完成 - 样本数: {len(norm_val_samples)}")

# ------------------------------------------------------------------
# 5. 模型前向传播：批量推理函数（复用逻辑）
# ------------------------------------------------------------------
def model_inference(norm_samples, batch_size=32):
    """
    模型批量推理函数
    :param norm_samples: 归一化后的样本 (N, tau, 4)
    :param batch_size: 批量大小
    :return: 归一化后的预测值
    """
    # 构建位置编码（与训练代码一致）
    t_pos = np.arange(tau)[:, np.newaxis]
    i_pos = np.arange(0, d_model, 2)
    div_term = np.exp(i_pos * (-np.log(10000.0) / d_model))
    P = np.zeros((tau, d_model))
    P[:, 0::2] = np.sin(t_pos * div_term)
    P[:, 1::2] = np.cos(t_pos * div_term)
    
    y_pred_norm_list = []
    for i in range(0, len(norm_samples), batch_size):
        X_batch = norm_samples[i:i+batch_size]
        B_actual = X_batch.shape[0]
        
        # 前向传播（完全匹配训练代码的Pre-LN逻辑）
        E_batch = X_batch @ W_e + b_e
        Z_batch = E_batch + P
        
        # Pre-LN + MHA
        LN_Z_batch = LayerNorm(Z_batch, gamma1, beta1)
        outs_MHA = MHA(LN_Z_batch, W_Q, W_K, W_V, W_O, h, d_K)
        res_1 = Z_batch + outs_MHA
        
        # Pre-LN + FFN
        LN_res1 = LayerNorm(res_1, gamma2, beta2)
        outs_FFN = FFN(LN_res1, W_1, b_1, W_2, b_2)
        res_2 = res_1 + outs_FFN
        
        # 平均池化 + 回归头
        final_repr = np.mean(res_2, axis=1)
        y_pred_norm = (final_repr @ W_pred + b_pred).squeeze(-1)
        
        y_pred_norm_list.append(y_pred_norm)
    
    return np.concatenate(y_pred_norm_list)

# ------------------------------------------------------------------
# 6. 训练集推理 + MAE计算
# ------------------------------------------------------------------
print("\n开始训练集推理...")
y_pred_train_norm = model_inference(norm_train_samples)
# 反归一化到原始尺度
y_pred_train_original = y_pred_train_norm * std_Y_train + mean_Y_train
y_true_train_original = train_labels_np

# 计算训练集MAE
mae_train_original = np.mean(np.abs(y_pred_train_original - y_true_train_original))
mae_train_norm = np.mean(np.abs(y_pred_train_norm - norm_train_labels))

# ------------------------------------------------------------------
# 7. 验证集推理 + MAE计算
# ------------------------------------------------------------------
print("\n开始验证集推理...")
y_pred_val_norm = model_inference(norm_val_samples)
# 反归一化到原始尺度
y_pred_val_original = y_pred_val_norm * std_Y_train + mean_Y_train
y_true_val_original = val_labels_np

# 计算验证集MAE
mae_val_original = np.mean(np.abs(y_pred_val_original - y_true_val_original))
mae_val_norm = np.mean(np.abs(y_pred_val_norm - norm_val_labels))

# ------------------------------------------------------------------
# 8. 打印评估结果
# ------------------------------------------------------------------
print("\n==================== 训练集评估结果 ====================")
print(f"原始尺度MAE: {mae_train_original:.4f} (PE的真实单位误差)")
print(f"归一化尺度MAE: {mae_train_norm:.6f} (对比训练时的MSE Loss)")
print(f"预测值均值: {y_pred_train_original.mean():.4f}, 真实值均值: {y_true_train_original.mean():.4f}")

print("\n==================== 验证集评估结果 ====================")
print(f"原始尺度MAE: {mae_val_original:.4f} (PE的真实单位误差)")
print(f"归一化尺度MAE: {mae_val_norm:.6f} (对比训练时的MSE Loss)")
print(f"预测值均值: {y_pred_val_original.mean():.4f}, 真实值均值: {y_true_val_original.mean():.4f}")

print("\n==================== 对比总结 ====================")
print(f"训练集MAE vs 验证集MAE (原始尺度): {mae_train_original:.4f} vs {mae_val_original:.4f}")
print(f"MAE差距: {mae_val_original - mae_train_original:.4f}")

# ------------------------------------------------------------------
# 9. 可视化：训练集（独立窗口1）
# ------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False

# 训练集可视化窗口
fig_train = plt.figure(figsize=(12, 10), num="训练集预测结果")
ax1_train = fig_train.add_subplot(211)
ax2_train = fig_train.add_subplot(212)

# 子图1：训练集真实值 vs 预测值曲线（取前500个样本）
sample_num = min(500, len(y_true_train_original))
x_axis = np.arange(sample_num)
ax1_train.plot(x_axis, y_true_train_original[:sample_num], label='真实值', color='blue', linewidth=1.5)
ax1_train.plot(x_axis, y_pred_train_original[:sample_num], label='预测值', color='red', linewidth=1, alpha=0.8)
ax1_train.set_title(f'训练集真实值 vs 预测值（前{sample_num}个样本） | MAE={mae_train_original:.4f}', fontsize=12)
ax1_train.set_xlabel('样本序号')
ax1_train.set_ylabel('PE值（原始尺度）')
ax1_train.legend()
ax1_train.grid(True, alpha=0.3)

# 子图2：训练集误差分布直方图
errors_train = y_pred_train_original - y_true_train_original
ax2_train.hist(errors_train, bins=50, color='green', alpha=0.7, edgecolor='black')
ax2_train.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='误差=0')
ax2_train.axvline(x=mae_train_original, color='orange', linestyle='--', linewidth=1.5, label=f'MAE={mae_train_original:.4f}')
ax2_train.set_title('训练集预测误差分布直方图', fontsize=12)
ax2_train.set_xlabel('预测误差（预测值 - 真实值）')
ax2_train.set_ylabel('样本数量')
ax2_train.legend()
ax2_train.grid(True, alpha=0.3)

# ------------------------------------------------------------------
# 10. 可视化：验证集（独立窗口2）
# ------------------------------------------------------------------
fig_val = plt.figure(figsize=(12, 10), num="验证集预测结果")
ax1_val = fig_val.add_subplot(211)
ax2_val = fig_val.add_subplot(212)

# 子图1：验证集真实值 vs 预测值曲线（取前500个样本）
sample_num_val = min(500, len(y_true_val_original))
x_axis_val = np.arange(sample_num_val)
ax1_val.plot(x_axis_val, y_true_val_original[:sample_num_val], label='真实值', color='blue', linewidth=1.5)
ax1_val.plot(x_axis_val, y_pred_val_original[:sample_num_val], label='预测值', color='red', linewidth=1, alpha=0.8)
ax1_val.set_title(f'验证集真实值 vs 预测值（前{sample_num_val}个样本） | MAE={mae_val_original:.4f}', fontsize=12)
ax1_val.set_xlabel('样本序号')
ax1_val.set_ylabel('PE值（原始尺度）')
ax1_val.legend()
ax1_val.grid(True, alpha=0.3)

# 子图2：验证集误差分布直方图
errors_val = y_pred_val_original - y_true_val_original
ax2_val.hist(errors_val, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax2_val.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='误差=0')
ax2_val.axvline(x=mae_val_original, color='orange', linestyle='--', linewidth=1.5, label=f'MAE={mae_val_original:.4f}')
ax2_val.set_title('验证集预测误差分布直方图', fontsize=12)
ax2_val.set_xlabel('预测误差（预测值 - 真实值）')
ax2_val.set_ylabel('样本数量')
ax2_val.legend()
ax2_val.grid(True, alpha=0.3)

# 保存图片（可选）
fig_train.savefig("./model/train_result.png", dpi=300, bbox_inches='tight')
fig_val.savefig("./model/val_result.png", dpi=300, bbox_inches='tight')
print("\n可视化结果已保存：")
print("- 训练集结果: ./model/train_result.png")
print("- 验证集结果: ./model/val_result.png")

# 显示所有窗口
plt.show()