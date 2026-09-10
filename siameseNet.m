% creating a siamese network 
% load data as an image datastore
imdsTrain = imageDatastore(data, ...
    'IncludeSubfolders',true, ...
    'LabelSource','none');

% Specify the labels manually by extracting the labels from the file names
files = imdsTrain.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),'_');
imdsTrain.Labels = categorical(labels);
idxs = randperm(numel(imdsTrain.Files),8);

% Display a random selection of the images.
for i = 1:numel(idxs)
    subplot(4,2,i)
    imshow(readimage(imdsTrain,idxs(i)))
    title(imdsTrain.Labels(idxs(i)), "Interpreter","none");
end

% specify options for training
numIterations = 10000;
miniBatchSize = 180;
learningRate = 6e-5;
trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];
gradDecay = 0.9;
gradDecaySq = 0.99;

% create a small representation set of five pairs of images
batchSize = 10;
[pairImage1,pairImage2,pairLabel] = getSiameseBatch(imdsTrain,batchSize);
% display generated images
for i = 1:batchSize    
    if pairLabel(i) == 1
        s = "similar";
    else
        s = "dissimilar";
    end
    subplot(2,5,i)   
    imshow([pairImage1(:,:,:,i) pairImage2(:,:,:,i)]);
    title(s)
end

% create subnetwork architecture
layers = [
    imageInputLayer([105 105 1],'Name','input1','Normalization','none')
    convolution2dLayer(10,64,'Name','conv1','WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool1')
    convolution2dLayer(7,128,'Name','conv2','WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool2')
    convolution2dLayer(4,128,'Name','conv3','WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool3')
    convolution2dLayer(5,256,'Name','conv4','WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')
    reluLayer('Name','relu4')
    fullyConnectedLayer(4096,'Name','fc1','WeightsInitializer','narrow-normal',...
    'BiasInitializer','narrow-normal')];

lgraph = layerGraph(layers);

% train the network with a custom training loop and enable automatic differentiation
dlnet = dlnetwork(lgraph);

% create the weights
% initialize the weights by sampling a random selection from a narrow normal 
% distribution with standard deviation of 0.01.
fcWeights = dlarray(0.01*randn(1,4096));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);

% create forward siamese function
function Y = forwardSiamese(dlnet,fcParams,dlX1,dlX2)
% forwardSiamese accepts the network and pair of training images, and returns a
% prediction of the probability of the pair being similar (closer to 1) or 
% dissimilar (closer to 0). Use forwardSiamese during training.

    % Pass the first image through the twin subnetwork
    F1 = forward(dlnet,dlX1);
    F1 = sigmoid(F1);
    
    % Pass the second image through the twin subnetwork
    F2 = forward(dlnet,dlX2);
    F2 = sigmoid(F2);
    
    % Subtract the feature vectors
    Y = abs(F1 - F2);
    
    % Pass the result through a fullyconnect operation
    Y = fullyconnect(Y,fcParams.FcWeights,fcParams.FcBias);
    
    % Convert to probability between 0 and 1.
    Y = sigmoid(Y);
end
    
% define model gradients function
function [gradientsSubnet,gradientsParams,loss] = modelGradients(dlnet,fcParams,dlX1,dlX2,pairLabels)
% The modelGradients function calculates the binary cross-entropy loss between the
% paired images and returns the loss and the gradients of the loss with respect to
% the network learnable parameters

    % Pass the image pair through the network 
    Y = forwardSiamese(dlnet,fcParams,dlX1,dlX2);
    
    % Calculate binary cross-entropy loss
    loss = binarycrossentropy(Y,pairLabels);
       
    % Calculate gradients of the loss with respect to the network learnable
    % parameters
    [gradientsSubnet,gradientsParams] = dlgradient(loss,dlnet.Learnables,fcParams);
end

function loss = binarycrossentropy(Y,pairLabels)
    % binarycrossentropy accepts the network's prediction Y, the true
    % label, and pairLabels, and returns the binary cross-entropy loss value.
    
    % Get precision of prediction to prevent errors due to floating
    % point precision    
    precision = underlyingType(Y);
      
    % Convert values less than floating point precision to eps.
    Y(Y < eps(precision)) = eps(precision);
    %convert values between 1-eps and 1 to 1-eps.
    Y(Y > 1 - eps(precision)) = 1 - eps(precision);
    
    % Calculate binary cross-entropy loss for each pair
    loss = -pairLabels.*log(Y) - (1 - pairLabels).*log(1 - Y);
    
    % Sum over all pairs in minibatch and normalize.
    loss = sum(loss)/numel(pairLabels);
end
% 
% % specify options for training
% numIterations = 10000;
% miniBatchSize = 180;
% learningRate = 6e-5;
% trailingAvgSubnet = [];
% trailingAvgSqSubnet = [];
% trailingAvgParams = [];
% trailingAvgSqParams = [];
% gradDecay = 0.9;
% gradDecaySq = 0.99;

% train using GPU
% executionEnvironment = "auto";
%  plots = "training-progress";

% Initialize the plot parameters for the training loss progress plot.
%plotRatio = 16/9;

%if plots == "training-progress"
 %   trainingPlot = figure;
  %  trainingPlot.Position(3) = plotRatio*trainingPlot.Position(4);
   % trainingPlot.Visible = 'on';
    
    %trainingPlotAxes = gca;
    
    %lineLossTrain = animatedline(trainingPlotAxes);
    %xlabel(trainingPlotAxes,"Iteration")
    %ylabel(trainingPlotAxes,"Loss")
    %title(trainingPlotAxes,"Loss During Training")
%end

% train model
% Loop over mini-batches.
%for iteration = 1:numIterations
    
    % Extract mini-batch of image pairs and pair labels
 %   [X1,X2,pairLabels] = getSiameseBatch(imdsTrain,miniBatchSize);
    
    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % 'SSCB' (spatial, spatial, channel, batch) for image data
  %  dlX1 = dlarray(single(X1),'SSCB');
   % dlX2 = dlarray(single(X2),'SSCB');
    
    % If training on a GPU, then convert data to gpuArray.
    %if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
     %   dlX1 = gpuArray(dlX1);
      %  dlX2 = gpuArray(dlX2);
    %end  
    
    % Evaluate the model gradients and the generator state using
    % dlfeval and the modelGradients function listed at the end of the
    % example.
    [gradientsSubnet, gradientsParams,loss] = dlfeval(@modelGradients,dlnet,fcParams,dlX1,dlX2,pairLabels);
    lossValue = double(gather(extractdata(loss)));
    
    % Update the Siamese subnetwork parameters.
    [dlnet,trailingAvgSubnet,trailingAvgSqSubnet] = ...
        adamupdate(dlnet,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);
    
    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = ...
        adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);
      
    % Update the training loss progress plot.
    if plots == "training-progress"
        addpoints(lineLossTrain,iteration,lossValue);
    end
    drawnow;
    
% load test data
imdsTest = imageDatastore(dataTest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','none');    

files = imdsTest.Files;
parts = split(files,filesep);
labels = join(parts(:,(end-2):(end-1)),'_');
imdsTest.Labels = categorical(labels);

% calculate accuracy
accuracy = zeros(1,5);
accuracyBatchSize = 150;

for i = 1:5
    
    % Extract mini-batch of image pairs and pair labels
    [XAcc1,XAcc2,pairLabelsAcc] = getSiameseBatch(imdsTest,accuracyBatchSize);
    
    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % 'SSCB' (spatial, spatial, channel, batch) for image data.
    dlXAcc1 = dlarray(single(XAcc1),'SSCB');
    dlXAcc2 = dlarray(single(XAcc2),'SSCB');
    
    % If using a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
       dlXAcc1 = gpuArray(dlXAcc1);
       dlXAcc2 = gpuArray(dlXAcc2);
    end    
    
    % Evaluate predictions using trained network
    dlY = predictSiamese(dlnet,fcParams,dlXAcc1,dlXAcc2);
   
    % Convert predictions to binary 0 or 1
    Y = gather(extractdata(dlY));
    Y = round(Y);
    
    % Compute average accuracy for the minibatch
    accuracy(i) = sum(Y == pairLabelsAcc)/accuracyBatchSize;
end

% Compute accuracy over all minibatches
% Do not the percentage suppressed, print immediately after running
averageAccuracy = mean(accuracy)*100

% display a test set of images and predictions
testBatchSize = 10;

[XTest1,XTest2,pairLabelsTest] = getSiameseBatch(imdsTest,testBatchSize);
    
% Convert test batch of data to dlarray. Specify the dimension labels
% 'SSCB' (spatial, spatial, channel, batch) for image data and 'CB' 
% (channel, batch) for labels
dlXTest1 = dlarray(single(XTest1),'SSCB');
dlXTest2 = dlarray(single(XTest2),'SSCB');


% train using GPU
 executionEnvironment = "auto";
 plots = "training-progress";

% Initialize the plot parameters for the training loss progress plot.
plotRatio = 16/9;

if plots == "training-progress"
    trainingPlot = figure;
    trainingPlot.Position(3) = plotRatio*trainingPlot.Position(4);
    trainingPlot.Visible = 'on';
    
    trainingPlotAxes = gca;
    
    lineLossTrain = animatedline(trainingPlotAxes);
    xlabel(trainingPlotAxes,"Iteration")
    ylabel(trainingPlotAxes,"Loss")
    title(trainingPlotAxes,"Loss During Training")
end

for iteration = 1:numIterations
    
    % Extract mini-batch of image pairs and pair labels
    [X1,X2,pairLabels] = getSiameseBatch(imdsTrain,miniBatchSize);
    
    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % 'SSCB' (spatial, spatial, channel, batch) for image data
    dlX1 = dlarray(single(X1),'SSCB');
    dlX2 = dlarray(single(X2),'SSCB');
    
    % If training on a GPU, then convert data to gpuArray.
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlX1 = gpuArray(dlX1);
        dlX2 = gpuArray(dlX2);
    end  
end



% If using a GPU, then convert data to gpuArray
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
   dlXTest1 = gpuArray(dlXTest1);
   dlXTest2 = gpuArray(dlXTest2);
end

% Calculate the predicted probability
dlYScore = predictSiamese(dlnet,fcParams,dlXTest1,dlXTest2);
YScore = gather(extractdata(dlYScore));

% Convert predictions to binary 0 or 1
YPred = round(YScore);    

% Extract data to plot
XTest1 = extractdata(dlXTest1);
XTest2 = extractdata(dlXTest2);

% Plot images with predicted label and predicted score
testingPlot = figure;
testingPlot.Position(3) = plotRatio*testingPlot.Position(4);
testingPlot.Visible = 'on';
    
for i = 1:numel(pairLabelsTest)
     
    if YPred(i) == 1
        predLabel = "similar";
    else
        predLabel = "dissimilar" ;
    end
    
    if pairLabelsTest(i) == YPred(i)
        testStr = "\bf\color{darkgreen}Correct\rm\newline";
        
    else
        testStr = "\bf\color{red}Incorrect\rm\newline";
    end
    
    subplot(2,5,i)        
    imshow([XTest1(:,:,:,i) XTest2(:,:,:,i)]);        
    
    title(testStr + "\color{black}Predicted: " + predLabel + "\newlineScore: " + YScore(i)); 
end



