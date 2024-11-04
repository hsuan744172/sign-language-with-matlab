% 設定測試檔案路徑
testFilePath = 'sign_mnist_test.csv';

% 讀取測試資料
testData = readmatrix(testFilePath);
X_test = testData(:, 2:end);
Y_test = testData(:, 1);

% 將標籤轉換為數字格式
Y_test_numeric = double(Y_test); 

% 計算每個標籤的出現次數
labelCounts = histcounts(Y_test_numeric, 'BinMethod', 'integers');

% 顯示每個標籤及其對應的數量
fprintf('Label counts:\n');
for i = 1:numel(labelCounts)
    if i <= numel(labelMapping)  % 確保不超出 labelMapping 的範圍
        labelText = labelMapping{i};  % 對應的字母
        count = labelCounts(i);        % 標籤數量
        fprintf('Label %s: %d\n', labelText, count);
    end
end 

% 讓使用者選擇要查看的標籤
selectedLabel = input('Enter the label number you want to view (1-24): ');

% 檢查選擇的標籤是否在有效範圍內
if selectedLabel < 1 || selectedLabel > numel(labelMapping)
    error('Invalid label number. Please enter a number between 1 and %d.', numel(labelMapping));
end

% 找到所選標籤的索引
indices = find(Y_test_numeric == selectedLabel);

% 顯示選擇的標籤的資料筆數
fprintf('Number of samples for label %s: %d\n', labelMapping{selectedLabel}, numel(indices));

% 如果有資料，讓使用者選擇資料的索引
if ~isempty(indices)
    sampleIndex = input('Enter the index of the sample you want to view (1 to %d): ', 's');
    sampleIndex = str2double(sampleIndex);

    % 檢查選擇的樣本索引是否有效
    if sampleIndex < 1 || sampleIndex > numel(indices)
        error('Invalid sample index. Please enter a number between 1 and %d.', numel(indices));
    end

    % 讀取並顯示所選樣本的圖像
    sampleImage = X_test(indices(sampleIndex), :);
    sampleImage = reshape(sampleImage, 28, 28); % 假設原圖是 28x28 的大小
    imshow(sampleImage, []);
    title(sprintf('Label: %s', labelMapping{selectedLabel}), 'FontSize', 14);
else
    disp('No samples found for the selected label.');
end