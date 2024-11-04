% 加載的模型文件
modelFilePath = 'hand_gesture_model.mat';
loadedData = load(modelFilePath);
net = loadedData.net;  % 將加載的模型賦值給 net 變量

% 使用 uigetfile 選擇圖片檔案
[imageFileName, imagePath] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', ...
    'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif, *.tiff)'; ...
    '*.*', 'All Files (*.*)'}, 'Select an Image');

if isequal(imageFileName, 0)
    disp('User canceled the file selection.');
else
    % 組合檔案路徑
    imageFilePath = fullfile(imagePath, imageFileName);
    
    % 讀取圖片
    originalImage = imread(imageFilePath);

    % 將圖片轉換為灰階
    grayImage = rgb2gray(originalImage); 

    % 調整圖片大小為 28x28 並進行標準化
    resizedImage = imresize(grayImage, [28 28]);
    normalizedImage = double(resizedImage) / 255.0;

    % 將圖片格式轉換為模型輸入格式
    inputImage = reshape(normalizedImage, [28, 28, 1, 1]);

    % 預測結果
    YPred = classify(net, inputImage);

    % 定義數字到字母的映射
    labelMapping = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', ...
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', ...
                    'U', 'V', 'W', 'X', 'Y'};

    % 獲取預測的標籤
    predictedLabel = labelMapping{double(YPred) + 1}; % +1 因為 MATLAB 的索引從 1 開始

    % 顯示原始圖片與預測結果
    figure;
    imshow(originalImage);
    title(['Predicted: ' predictedLabel], 'FontSize', 14, 'Interpreter', 'none');
end
