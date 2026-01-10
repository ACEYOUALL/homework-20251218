clear; clc; close all;

fprintf('正在加载数据...\n');
TrainT = readtable('Training_Set.csv');
TestT  = readtable('Testing_Set.csv');

TrainX = table2array(TrainT(:,1:4));
TrainY = table2array(TrainT(:,5));
TestX  = table2array(TestT(:,1:4));

[TrainX, Mu, Sigma] = zscore(TrainX);
TestX = (TestX - Mu) ./ Sigma;

rng(0); 
Cv = cvpartition(size(TrainX,1), 'HoldOut', 0.3);
TrainIdx = training(Cv);
ValIdx   = test(Cv);

XTrain = TrainX(TrainIdx,:)';
YTrain = TrainY(TrainIdx);
XVal   = TrainX(ValIdx,:)';
YVal   = TrainY(ValIdx);
XTest  = TestX';

XTrain3D = reshape(XTrain, 4, 1, []);
XVal3D   = reshape(XVal,   4, 1, []);
XTest3D  = reshape(XTest,  4, 1, []);

fprintf('训练数据维度: %dx%dx%d\n', size(XTrain3D));
fprintf('验证数据维度: %dx%dx%d\n', size(XVal3D));
fprintf('测试数据维度: %dx%dx%d\n', size(XTest3D));

fprintf('构建 Transformer 近似模型...\n');
LayersTrans = [
    sequenceInputLayer(4, 'Normalization','none')
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

fprintf('构建 LSTM 对比模型...\n');
LayersLstm = [
    sequenceInputLayer(4, 'Normalization','none')
    lstmLayer(50, 'OutputMode','last')
    fullyConnectedLayer(1)
    regressionLayer];

Options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationData', {XVal3D, YVal}, ...
    'ValidationFrequency', 10, ...
    'ValidationPatience', 15, ...
    'ExecutionEnvironment', 'auto');

fprintf('训练 Transformer 近似模型...\n');
NetTrans = trainNetwork(XTrain3D, YTrain, LayersTrans, Options);

fprintf('训练 LSTM 模型...\n');
NetLstm = trainNetwork(XTrain3D, YTrain, LayersLstm, Options);

PredTrans = predict(NetTrans, XTest3D);
PredLstm  = predict(NetLstm,  XTest3D);

writematrix(PredTrans, 'Prediction_Transformer.csv');
writematrix(PredLstm,  'Prediction_LSTM.csv');
fprintf('预测结果已保存。\n');

figure;
T = 1:min(200, length(PredTrans));
plot(T, PredTrans(T), 'b-', 'LineWidth', 1.5); hold on;
plot(T, PredLstm(T),  'r--', 'LineWidth', 1.5);
xlabel('Time (hour)'); ylabel('Predicted PE (MW)');
title('Prediction Comparison: Transformer vs LSTM (First 200 Hours)');
legend('Transformer (Approx)', 'LSTM');
grid on;