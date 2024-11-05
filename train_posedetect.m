% 設定檔案路徑
trainFilePath = 'sign_mnist_train.csv';

% 讀取並處理訓練和測試資料
trainData = readmatrix(trainFilePath);

% 提取標籤和圖像資料
X = trainData(:, 2:end);
Y = trainData(:, 1);

% 使用隨機種子以確保分割一致性
rng('default'); % 設定隨機種子，確保每次執行得到相同的結果

% 將資料分為訓練集和驗證集（80%訓練，20%驗證）
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 創建交叉驗證物件
idx = cv.test; % 獲取測試的邏輯索引

% 檢查訓練集和驗證集的分佈，確保隨機分配且樣本數量合理
fprintf('訓練樣本數: %d, 驗證樣本數: %d\n', sum(~idx), sum(idx));

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

layers = [
    imageInputLayer([28 28 1], 'Normalization', 'none')
    convolution2dLayer(3, 64, 'Padding', 'same') % Conv2D with 64 filters
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % MaxPooling2D
    convolution2dLayer(3, 64, 'Padding', 'same') % Second Conv2D
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % Second MaxPooling2D
    convolution2dLayer(3, 64, 'Padding', 'same') % Third Conv2D
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % Third MaxPooling2D
    flattenLayer % Flatten
    fullyConnectedLayer(128) % Dense layer with 128 units
    reluLayer
    dropoutLayer(0.2) % Dropout
    fullyConnectedLayer(24) % Output layer with 24 units
    softmaxLayer
    classificationLayer
];

% 訓練選項
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 訓練模型
net = trainNetwork(X_train, Y_train, layers, options);

% 保存模型
modelFilePath = 'hand_gesture_model.mat';
save(modelFilePath, 'net');
