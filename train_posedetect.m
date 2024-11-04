% 設定檔案路徑
trainFilePath = 'sign_mnist_train.csv';
testFilePath = 'sign_mnist_test.csv';

% 讀取並處理訓練和測試資料
trainData = readmatrix(trainFilePath);
testData = readmatrix(testFilePath);

% 提取標籤和圖像資料
X_train = trainData(:, 2:end);
Y_train = trainData(:, 1);
X_test = testData(:, 2:end);
Y_test = testData(:, 1);

% 將圖像大小重新調整為 28x28 並進行標準化
X_train = reshape(X_train', 28, 28, 1, []) / 255.0;
X_test = reshape(X_test', 28, 28, 1, []) / 255.0;

% 將標籤轉換為分類格式
Y_train = categorical(Y_train);
Y_test = categorical(Y_test);

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
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_test, Y_test}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 訓練模型
net = trainNetwork(X_train, Y_train, layers, options);
% 保存模型
modelFilePath = 'hand_gesture_model.mat';  % 指定保存的文件名
save(modelFilePath, 'net');  % 將訓練好的模型保存到文件中

%https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data資料集來源


