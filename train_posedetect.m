% 設定檔案路徑
trainFilePath = 'sign_mnist_train.csv';

% 讀取並處理訓練和測試資料
trainData = readmatrix(trainFilePath);

% 提取標籤和圖像資料
X = trainData(:, 2:end);
Y = trainData(:, 1);

% 將資料分為訓練集和驗證集（80%訓練，20%驗證）
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 創建交叉驗證物件
idx = cv.test; % 獲取測試的邏輯索引

% 分割訓練和驗證資料
X_train = X(~idx, :); % 訓練集
Y_train = Y(~idx);     % 訓練集標籤
X_val = X(idx, :);     % 驗證集
Y_val = Y(idx);        % 驗證集標籤

% 將圖像大小重新調整為 28x28 並進行標準化
X_train = reshape(X_train', 28, 28, 1, []) / 255.0;
X_val = reshape(X_val', 28, 28, 1, []) / 255.0;

% 將標籤轉換為分類格式
Y_train = categorical(Y_train);
Y_val = categorical(Y_val);

% 建立模型
layers = [
    imageInputLayer([28 28 1], 'Normalization', 'none')

    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
    batchNormalizationLayer
    dropoutLayer(0.2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    batchNormalizationLayer
    dropoutLayer(0.2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    batchNormalizationLayer
    dropoutLayer(0.2)

    % 全連接層
    fullyConnectedLayer(512)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.25)
    fullyConnectedLayer(126)
    reluLayer
    fullyConnectedLayer(24)
    softmaxLayer
    classificationLayer
];

% 訓練選項
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_val, Y_val}, ... % 使用驗證集
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 訓練模型
net = trainNetwork(X_train, Y_train, layers, options);

% 保存模型
modelFilePath = 'hand_gesture_model.mat';  % 指定保存的文件名
save(modelFilePath, 'net');  % 將訓練好的模型保存到文件中
