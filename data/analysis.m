%% 读取数据集

data = readtable('training_set.csv');

AT = data.AT;
EV = data.EV;
AP = data.AP;
RH = data.RH;
PE = data.PE;

%% 延迟嵌入

tau_embed = 3;
X_embed = [PE(1:end-tau_embed+1), ...
           PE(2:end-tau_embed+2), ...
           PE(3:end)];
plot3(X_embed(:,1), X_embed(:,2), X_embed(:,3), '.');

%% ACF分析

lags = 0:1000;

figure('Position', [100, 100, 1200, 800]);

subplot(3,2,1); autocorr(AT, 'NumLags', max(lags)); title('ACF of AT');
subplot(3,2,2); autocorr(EV, 'NumLags', max(lags)); title('ACF of EV');
subplot(3,2,3); autocorr(AP, 'NumLags', max(lags)); title('ACF of AP');
subplot(3,2,4); autocorr(RH, 'NumLags', max(lags)); title('ACF of RH');
subplot(3,2,5); autocorr(PE, 'NumLags', max(lags)); title('ACF of PE');
sgtitle('Autocorrelation Function of All Variables');