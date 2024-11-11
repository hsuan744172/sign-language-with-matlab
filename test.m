% 這個檔案用來顯示CSV原圖大小
% 設定測試檔案路徑
testFilePath = 'sign_mnist_test.csv';

% 讀取測試資料
testData = readmatrix(testFilePath);

% 查看資料的大小
[numRows, numCols] = size(testData);
fprintf('資料集大小: %d 筆資料, 每筆資料有 %d 個欄位\n', numRows, numCols);

% 查看第一筆資料的前幾個值
disp('第一筆資料的前10個數值:');
disp(testData(1, 1:10));

% 檢查圖像大小
imageData = testData(1, 2:end); % 排除標籤欄位
numPixels = length(imageData);
imageSize = sqrt(numPixels); % 假設圖像是正方形
fprintf('圖像大小: %d x %d 像素\n', imageSize, imageSize);

% 顯示第一筆圖像
imageMatrix = reshape(imageData, [imageSize, imageSize]); % 重塑為二維圖像
imshow(imageMatrix, []);
title('第一筆測試圖像');
