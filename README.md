# Transfer-Learning-Using-AlexNet
- AlexNet 컨벌루션 신경망이 새로운 영상 모음에 대해 분류를 수행하도록 미세 조정하는 방법을 보여줍니다.

1백만 개가 넘는 영상에 대해 훈련된 AlexNet은 영상을 키보드, 커피 머그잔, 연필, 각종 동물 등 1,000가지 사물 범주로 분류할 수 있습니다. 이 신경망은 다양한 영상을 대표하는 다양한 특징을 학습했습니다.
사전 훈련된 신경망을 새로운 작업을 학습하기 위한 출발점으로 사용할 수 있습니다. 전이 학습으로 신경망을 미세 조정하는 것은 무작위로 초기화된 가중치를 사용하여 신경망을 처음부터 훈련시키는 것보다 일반적으로 훨씬 더 빠르고 쉽습니다.

### 데이터 불러오기
- 새 이미지의 압축을 풀고 영상 데이터저장소로 불러옵니다.
```c
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ... %'IncludeSubfolders'-하위폴더 포함 플래그
    'LabelSource','foldernames'); %'LabelSource'-레이블 데이터를 제공하는 소스
```

- 데이터를 훈련와 검증 데이터 세트로 나눕니다.
영상의 70%를 훈련용으로 사용하고 30%를 검증용으로 사용합니다.
splitEachLabel은 images 데이터저장소를 2개의 새로운 데이터저장소로 분할합니다.
```c
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
```

- 이제 데이터 세트에는 55개의 훈련 영상과 20개의 검증 영상이 포함됩니다.
- 샘플 영상 몇개를 표시합니다.
```c
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
```

![untitled2](https://user-images.githubusercontent.com/86040099/127144093-85d65cc2-06f5-47a5-9bb4-95372fb0fb55.jpg)

### 사전 훈련된 신경망 불러오기
- 사전 훈련된 AlexNet 신경망을 불러옵니다.
```c
net = alexnet;
```
![화면 캡처 2021-07-27 201505](https://user-images.githubusercontent.com/86040099/127145058-8ad5f7cd-4550-4cec-a66a-f05111d2cdb5.jpg)

- 첫 번째 계층인 영상 입력 계층에 입력되는 영상은 크기가 227x227x3이어야 합니다.
```c
inputSize = net.Layers(1).InputSize
```

### 마지막 계층 바꾸기
- 사전 훈련된 신경망의 마지막 세 계층은 1000개의 클래스에 대해 구성되어 있기 때문에 이 세 계층을 새로운 훈련 데이터에 맞게 조정합니다.
- 마지막 3개의 계층을 제외한 계층을 추출합니다.
```c
layersTransfer = net.Layers(1:end-3);
```
마지막 3개의 계층을 새로운 계층으로 바꿔줍니다. 이때, 완전 연결 계층이 새로운 데이터의 클래스 개수와 동일한 크기를 갖도록 설정합니다.
```c
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
```

### 신경망 훈련시키기

```c
pixelRange = [-30 30]; %픽셀 크기 지정
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ... %무작위 반사
    'RandXTranslation',pixelRange, ... %가로 평행 이동 범위
    'RandYTranslation',pixelRange); %세로 평행 이동 범위
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter); %'DataAugmentation'-입력 영상에 적용할 전처리
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
```

- 훈련 옵션을 지정합니다.
```c
options = trainingOptions('sgdm', ... %SGDM-모멘텀을 사용한 확률적 경사하강법
    'MiniBatchSize',10, ... %미니 배치의 크기
    'MaxEpochs',6, ... %최대 Epoch 횟수
    'InitialLearnRate',1e-4, ... %초기 학습률
    'Shuffle','every-epoch', ... %데이터 섞기 옵션
    'ValidationData',augimdsValidation, ... %훈련 중에 검증에 사용할 데이터
    'ValidationFrequency',3, ... %신경망 검증 빈도
    'Verbose',false, ...
    'Plots','training-progress');
```
*Epoch 1회는 훈련 알고리즘이 전체 훈련 세트를 완전히 한 번 통과하는 것을 의미합니다.*

- 신경망을 훈련시킵니다.
```c
netTransfer = trainNetwork(augimdsTrain,layers,options);
```

![화면 캡처 2021-07-27 201241](https://user-images.githubusercontent.com/86040099/127149454-b34fdcce-2879-4cca-baab-17e3c6689e20.jpg)


### 검증 영상 분류하기
- 훈련시킨 신경망을 사용하여 검증 영상을 분류합니다.
```c
[YPred,scores] = classify(netTransfer,augimdsValidation);
```
*훈련된 신경망 netTransfer를 사용하여 augimdsValidation의 이미지 데이터에 대한 클래스 레이블을 예측합니다.*

-4개의 샘플 이미지를 예측된 레이블과 함께 표시합니다.
```c
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
```

![untitled](https://user-images.githubusercontent.com/86040099/127149115-7fd5514e-6dae-4258-b371-e0de927a0d92.jpg)

-검증 세트에 대한 분류 정확도를 계산합니다.
```c
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
```

![화면 캡처 2021-07-27 205400](https://user-images.githubusercontent.com/86040099/127149288-5746eac4-1e01-4608-bdba-72451029c2d7.jpg)
