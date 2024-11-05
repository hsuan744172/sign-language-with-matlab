% 設定測試檔案路徑
testFilePath = 'sign_mnist_test.csv';

% 讀取測試資料
testData = readmatrix(testFilePath);

% 提取測試圖像資料和標籤
X_test = testData(:, 2:end);
Y_test = testData(:, 1);

% 將圖像大小重新調整為 28x28 並進行標準化
X_test = reshape(X_test', 28, 28, 1, []) / 255.0;

% 將標籤轉換為分類格式
Y_test = categorical(Y_test);

% 加載已訓練的模型
modelFilePath = 'hand_gesture_model.mat';
load(modelFilePath, 'net');

% 使用模型進行預測
Y_pred = classify(net, X_test);

% 計算準確度
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('測試資料準確度: %.2f%%\n', accuracy * 100);

%epoch 2 測試資料準確度: 90.59%
%epoch 3 測試資料準確度: 93.40%