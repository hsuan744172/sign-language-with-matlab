% 加載的模型文件
modelFilePath = 'hand_gesture_model.mat';
loadedData = load(modelFilePath);
net = loadedData.net;  % 將加載的模型賦值給 net 變量

% 設定測試檔案路徑
testFilePath = 'sign_mnist_test.csv';

% 讀取測試資料
testData = readmatrix(testFilePath);
X_test = testData(:, 2:end);
Y_test = testData(:, 1);

% 將圖像大小重新調整為 28x28 並進行標準化
X_test = reshape(X_test', 28, 28, 1, []) / 255.0;

% 將標籤轉換為分類格式
Y_test = categorical(Y_test);

% 隨機挑選五個樣本
numSamples = 5;  % 要挑選的樣本數
randomIndices = randperm(size(X_test, 4), numSamples);  % 隨機選取索引

% 預測和顯示結果
YPred = classify(net, X_test(:, :, 1, randomIndices));
accuracy = sum(YPred == Y_test(randomIndices)) / numel(randomIndices);


% 定義數字到字母的映射
labelMapping = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', ...
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', ...
                'U', 'V', 'W', 'X', 'Y'};

% 可視化隨機挑選的預測結果
figure;
for i = 1:numSamples
    % 顯示原始圖像
    subplot(2, numSamples, i);
    originalImage = reshape(testData(randomIndices(i), 2:end)', 28, 28);  % 從 CSV 中獲取原始圖像
    imshow(originalImage, []);
    title('Original Image', 'FontSize', 10, 'Interpreter', 'none');

    % 顯示預測圖像
    subplot(2, numSamples, i + numSamples);
    predictedImage = X_test(:, :, 1, randomIndices(i));
    imshow(predictedImage, []);
    
    % 在圖片上方顯示實際和預測標籤
    actualLabel = Y_test(randomIndices(i));  % 這裡是數字
    predictedLabel = YPred(i);  % 這裡是數字
    
   
    actualLabelText = labelMapping{double(actualLabel)};
    predictedLabelText = labelMapping{double(predictedLabel)}; 
    
    titleText = ['Actual: ' actualLabelText, newline, 'Predicted: ' predictedLabelText];
    title(titleText, 'FontSize', 10, 'Interpreter', 'none');  % 調整字體大小
end
