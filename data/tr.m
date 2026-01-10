%% 联合循环电厂净电能输出预测（MATLAB R2021b 兼容 - 修正版）
% 该程序使用深度学习模型预测电厂每小时净电能输出(PE)
% 由于MATLAB R2021b不支持原生Transformer层，我们采用全连接网络近似Transformer的特征交互能力
% 同时构建LSTM模型进行对比分析
clear; clc; close all;

%% 1. 加载数据
% 读取训练集和测试集
fprintf('正在加载数据...\n');
trainT = readtable('training_set.csv');  % 包含特征(AT,EV,AP,RH)和标签(PE)
testT  = readtable('testing_set.csv');   % 仅包含特征(AT,EV,AP,RH)，无PE标签

% 提取特征和标签
% 训练集: 前4列为特征(AT,EV,AP,RH)，第5列为标签(PE)
% 测试集: 仅有4个特征列
trainX = table2array(trainT(:,1:4));  % N×4 矩阵，N为样本数量
trainY = table2array(trainT(:,5));    % N×1 向量
testX  = table2array(testT(:,1:4));   % M×4 矩阵，M为测试样本数量

%% 2. 数据标准化（Z-score标准化）
% 重要步骤：消除不同特征间的量纲差异，加速模型收敛
% 标准化公式: z = (x - μ) / σ
[trainX, mu, sigma] = zscore(trainX);  % 对训练集标准化，并记录均值和标准差
testX = (testX - mu) ./ sigma;         % 使用训练集的统计量标准化测试集，避免数据泄露

%% 3. 划分训练/验证集
% 固定随机种子以确保结果可复现
rng(0); 
% 将训练数据划分为70%训练集和30%验证集
cv = cvpartition(size(trainX,1), 'HoldOut', 0.3);
trainIdx = training(cv);  % 训练集索引
valIdx   = test(cv);      % 验证集索引

% 提取训练集和验证集
XTrain = trainX(trainIdx,:)';  % 4×Ntr 矩阵 (4个特征，Ntr个训练样本)
YTrain = trainY(trainIdx);    % Ntr×1 向量
XVal   = trainX(valIdx,:)';    % 4×Nval 矩阵 (4个特征，Nval个验证样本)
YVal   = trainY(valIdx);       % Nval×1 向量
XTest  = testX';               % 4×M 矩阵 (4个特征，M个测试样本)

%% 4. 转换为3D序列格式 (C×S×N)
% MATLAB的sequenceInputLayer要求3D输入格式(C×S×N):
%   C = 通道数 = 4 (特征数量)
%   S = 序列长度 = 1 (每条样本被视为长度为1的序列)
%   N = 样本数量
XTrain3D = reshape(XTrain, 4, 1, []);
XVal3D   = reshape(XVal,   4, 1, []);
XTest3D  = reshape(XTest,  4, 1, []);

% 打印维度信息用于调试
fprintf('训练数据维度: %dx%dx%d\n', size(XTrain3D));
fprintf('验证数据维度: %dx%dx%d\n', size(XVal3D));
fprintf('测试数据维度: %dx%dx%d\n', size(XTest3D));

%% 5. 构建 "Transformer 近似" 模型
% 由于MATLAB R2021b不支持原生的Transformer层，我们使用多层全连接网络模拟Transformer的特征交互能力
% 这是一种简化但有效的近似，尤其适用于特征相对较少(4个)的预测任务
fprintf('构建 Transformer 近似模型...\n');
layers_trans = [
    sequenceInputLayer(4, 'Normalization','none')  % 输入层，4个特征，禁用内置标准化(已手动标准化)
    fullyConnectedLayer(64)  % 第一个全连接层，64个神经元
    reluLayer                % ReLU激活函数，引入非线性
    fullyConnectedLayer(32)  % 第二个全连接层，32个神经元
    reluLayer                % ReLU激活函数
    fullyConnectedLayer(1)   % 输出层，1个神经元(预测PE)
    regressionLayer];        % 回归层，用于连续值预测

%% 6. 构建 LSTM 对比模型
% LSTM是时间序列预测的经典模型，用于与Transformer近似模型进行对比
fprintf('构建 LSTM 对比模型...\n');
layers_lstm = [
    sequenceInputLayer(4, 'Normalization','none')  % 输入层，4个特征
    lstmLayer(50, 'OutputMode','last') % LSTM层，50个隐藏单元，只输出序列最后一个时间步
    fullyConnectedLayer(1)   % 输出层
    regressionLayer];        % 回归层

%% 7. 训练选项
% 配置训练参数
options = trainingOptions('adam', ...          % 使用Adam优化器
    'MaxEpochs', 100, ...                       % 最大训练轮次
    'MiniBatchSize', 64, ...                    % 小批量大小
    'InitialLearnRate', 1e-3, ...               % 初始学习率
    'Plots', 'training-progress', ...           % 显示训练进度图
    'Verbose', false, ...                       % 不显示详细训练信息
    'ValidationData', {XVal3D, YVal}, ...       % 验证集
    'ValidationFrequency', 10, ...              % 每10次迭代验证一次
    'ValidationPatience', 15, ...               % 验证损失15次迭代没有改善则停止训练(早停)
    'ExecutionEnvironment', 'auto');           % 自动选择CPU/GPU

%% 8. 训练 Transformer 近似模型
fprintf('训练 Transformer 近似模型...\n');
net_trans = trainNetwork(XTrain3D, YTrain, layers_trans, options);

%% 9. 训练 LSTM 模型
fprintf('训练 LSTM 模型...\n');
net_lstm = trainNetwork(XTrain3D, YTrain, layers_lstm, options);

%% 10. 预测
% 使用训练好的模型对测试集进行预测
pred_trans = predict(net_trans, XTest3D);  % Transformer模型预测
pred_lstm  = predict(net_lstm,  XTest3D);  % LSTM模型预测

%% 11. 保存结果
% 将预测结果保存为CSV文件
writematrix(pred_trans, 'Prediction_Transformer.csv');
writematrix(pred_lstm,  'Prediction_LSTM.csv');
fprintf('预测结果已保存。\n');

%% 12. 可视化（前 200 小时）
% 绘制前200小时的预测结果对比
figure;
t = 1:min(200, length(pred_trans));  % 取前200小时或全部(如果少于200)
plot(t, pred_trans(t), 'b-', 'LineWidth', 1.5); hold on;  % Transformer预测 (蓝色实线)
plot(t, pred_lstm(t),  'r--', 'LineWidth', 1.5);          % LSTM预测 (红色虚线)
xlabel('Time (hour)'); ylabel('Predicted PE (MW)');
title('Prediction Comparison: Transformer vs LSTM (First 200 Hours)');
legend('Transformer (Approx)', 'LSTM');
grid on;