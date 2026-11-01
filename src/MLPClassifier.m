%% MLPClassifier.m
% Train and evaluate a Multi-Layer Perceptron (MLP)
% for sunglasses detection using the CMU face dataset.

clear; clc; close all;


%% 1. Define project paths and read the prepared dataset

rng(42);

% Directory containing this script (src).
srcRoot = fileparts(mfilename("fullpath"));

% Project root is the parent directory of src.
projectRoot = fileparts(srcRoot);

% Directory for saving result visualizations.
imagesRoot = fullfile(projectRoot, "images");

% Create the images directory if it does not exist.
if ~isfolder(imagesRoot)
    mkdir(imagesRoot);
end


% Prepared dataset directory.
datasetRoot = fullfile(projectRoot, "dataset");

% Verify that the dataset exists.
assert(isfolder(datasetRoot), ...
    "Dataset directory not found: %s. Run prepare_dataset.m first.", ...
    datasetRoot);

% Load images and assign labels based on folder names.
imds = imageDatastore(datasetRoot, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");



% Verify that exactly two classes exist.
classNames = categories(imds.Labels);

assert(numel(classNames) == 2, ...
    "The dataset must contain exactly two class folders.");

assert(any(string(classNames) == "sunglasses"), ...
    "The dataset must contain a 'sunglasses' class.");

disp("Full dataset:");
disp(countEachLabel(imds));

%% 2. Create the held-out test set

% Use the same seed and split procedure as the SVM experiment.
[imdsTrain, imdsTest] = splitEachLabel( ...
    imds, 0.80, "randomized");

disp("Training data:");
disp(countEachLabel(imdsTrain));

disp("Test data:");
disp(countEachLabel(imdsTest));


%% 3. Define image dimensions and extract features

imageHeight = 120;
imageWidth = 128;

XTrain = imagesToFeatures( ...
    imdsTrain, imageHeight, imageWidth);

XTest = imagesToFeatures( ...
    imdsTest, imageHeight, imageWidth);

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% Encode sunglasses as 1 and no_sunglasses as 0.
YTrainBinary = single(string(YTrain) == "sunglasses");
YTestBinary = single(string(YTest) == "sunglasses");

inputSize = size(XTrain, 2);

fprintf("Number of training images: %d\n", size(XTrain, 1));
fprintf("Number of test images: %d\n", size(XTest, 1));
fprintf("Number of input features: %d\n", inputSize);


%% 4. Define the MLP architecture and hyperparameters

hiddenSize = 64;

learningRate = 0.01;
batchSize = 32;
maxEpochs = 3000;

lambda = 0.0001;

patience = 25;

fprintf("\nMLP Architecture:\n");
fprintf("Input neurons:  %d\n", inputSize);
fprintf("Hidden neurons: %d\n", hiddenSize);
fprintf("Output neurons: 1\n");


%% 5. Create a validation set from the training data

% The held-out test set is not used for model selection.
rng(123);

validationPartition = cvpartition( ...
    YTrain, "Holdout", 0.20);

trainingIndices = training(validationPartition);
validationIndices = test(validationPartition);

XSubTrain = XTrain(trainingIndices, :);
YSubTrain = YTrainBinary(trainingIndices);

XValidation = XTrain(validationIndices, :);
YValidation = YTrainBinary(validationIndices);

fprintf("\nSub-training images: %d\n", size(XSubTrain, 1));
fprintf("Validation images:   %d\n", size(XValidation, 1));


%% 6. Determine the best number of epochs using early stopping

[~, validationHistory, bestEpoch] = trainMLP( ...
    XSubTrain, YSubTrain, ...
    hiddenSize, maxEpochs, batchSize, ...
    learningRate, lambda, ...
    42, XValidation, YValidation, patience);

fprintf("\nSelected number of epochs: %d\n", bestEpoch);

figure("Name", "MLP Validation Loss");

plot(1:numel(validationHistory), ...
    validationHistory, "LineWidth", 1.5);

xlabel("Epoch");
ylabel("Binary Cross-Entropy Loss");

title("MLP Validation Loss");
grid on;

exportgraphics(gcf, ...
    fullfile(imagesRoot, "MLP_validation_loss.png"), ...
    Resolution=300);


%% 7. Train the final MLP using the complete training set

% Reinitialize the network and train on all training samples.
% The number of epochs was selected using the validation set.

[finalMLP, ~, ~] = trainMLP( ...
    XTrain, YTrainBinary, ...
    hiddenSize, bestEpoch, batchSize, ...
    learningRate, lambda, ...
    42, [], [], patience);

fprintf("\nFinal MLP training completed.\n");


%% 8. Evaluate the final MLP on the held-out test set

% The output is a probability-like score between 0 and 1.
testScores = predictMLP(finalMLP, XTest);

% A threshold of 0.5 determines the predicted class.
testPredictedBinary = testScores >= 0.5;

testPredictedLabels = categorical( ...
    double(testPredictedBinary), ...
    [0 1], ...
    ["no_sunglasses", "sunglasses"]);

testAccuracy = mean( ...
    double(testPredictedBinary) == double(YTestBinary));

fprintf("\nHeld-Out Test Results:\n");
fprintf("Test accuracy: %.2f%%\n", 100 * testAccuracy);


%% 9. Calculate additional evaluation metrics

% Positive class: sunglasses.
% Negative class: no_sunglasses.

yTrue = logical(YTestBinary);
yPred = logical(testPredictedBinary);

TP = sum(yTrue & yPred);
TN = sum(~yTrue & ~yPred);
FP = sum(~yTrue & yPred);
FN = sum(yTrue & ~yPred);

accuracy = (TP + TN) / numel(yTrue);

precision = TP / max(TP + FP, 1);
recall = TP / max(TP + FN, 1);

specificity = TN / max(TN + FP, 1);

f1Score = 2 * precision * recall / ...
    max(precision + recall, eps);

balancedAccuracy = (recall + specificity) / 2;

fprintf("\nAdditional Test Metrics:\n");

fprintf("True Positives:    %d\n", TP);
fprintf("True Negatives:    %d\n", TN);
fprintf("False Positives:   %d\n", FP);
fprintf("False Negatives:   %d\n", FN);

fprintf("Accuracy:          %.4f\n", accuracy);
fprintf("Precision:         %.4f\n", precision);
fprintf("Recall:            %.4f\n", recall);
fprintf("Specificity:       %.4f\n", specificity);
fprintf("F1-score:          %.4f\n", f1Score);
fprintf("Balanced accuracy: %.4f\n", balancedAccuracy);


%% 10. Display the confusion matrix

figure("Name", "MLP Test-Set Confusion Matrix");

confusionchart(YTest, testPredictedLabels);

title("MLP: Sunglasses Detection");

exportgraphics(gcf, ...
    fullfile(imagesRoot, "MLP_confusion_matrix.png"), ...
    Resolution=300);

%% 11. Display random test images and their predictions

numberOfImagesToShow = min(10, numel(imdsTest.Files));

rng(100);

testImageIndices = randperm( ...
    numel(imdsTest.Files), numberOfImagesToShow);

numberOfColumns = 5;
numberOfRows = ceil(numberOfImagesToShow / numberOfColumns);

figure("Name", "MLP Test Images and Predictions");

tiledlayout(numberOfRows, numberOfColumns, ...
    Padding="compact", ...
    TileSpacing="compact");

for imageNumber = 1:numberOfImagesToShow

    testIndex = testImageIndices(imageNumber);

    image = readimage(imdsTest, testIndex);

    trueLabel = YTest(testIndex);
    predictedLabel = testPredictedLabels(testIndex);

    nexttile;
    imshow(image, []);

    if trueLabel == predictedLabel
        titleColor = [0, 0.5, 0];
    else
        titleColor = [0.8, 0, 0];
    end

    title(sprintf("True: %s\nPredicted: %s", ...
        string(trueLabel), string(predictedLabel)), ...
        Color=titleColor, ...
        Interpreter="none");
end

sgtitle("MLP Test Predictions: Green = Correct, Red = Incorrect");
exportgraphics(gcf, ...
    fullfile(imagesRoot, "MLP_test_predictions.png"), ...
    Resolution=300);


%% 12. Display misclassified test images

misclassifiedIndices = find( ...
    testPredictedLabels ~= YTest);

fprintf("\nMisclassified images: %d / %d\n", ...
    numel(misclassifiedIndices), numel(YTest));

if ~isempty(misclassifiedIndices)

    numberOfErrorsToShow = min(12, ...
        numel(misclassifiedIndices));

    figure("Name", "MLP Misclassified Images");

    tiledlayout(ceil(numberOfErrorsToShow / 4), 4, ...
        Padding="compact", ...
        TileSpacing="compact");

    for k = 1:numberOfErrorsToShow

        testIndex = misclassifiedIndices(k);

        image = readimage(imdsTest, testIndex);

        nexttile;
        imshow(image, []);

        title(sprintf("True: %s\nPred: %s", ...
            string(YTest(testIndex)), ...
            string(testPredictedLabels(testIndex))), ...
            Interpreter="none");
    end

    sgtitle("MLP Misclassified Test Images");
    exportgraphics(gcf, ...
    fullfile(imagesRoot, "MLP_misclassified_images.png"), ...
    Resolution=300);

end


%% 13. Save the trained model

% Directory for trained models.
modelsRoot = fullfile(projectRoot, "models");

% Create the models directory if it does not exist.
if ~isfolder(modelsRoot)
    mkdir(modelsRoot);
end

% Path to the saved MLP model.
modelFile = fullfile(modelsRoot, "trainedMLPModel.mat");

% Save the trained model and its hyperparameters.
save(modelFile, ...
    "finalMLP", ...
    "hiddenSize", ...
    "learningRate", ...
    "batchSize", ...
    "lambda", ...
    "bestEpoch", ...
    "imageHeight", ...
    "imageWidth");

fprintf("\nMLP model saved to:\n%s\n", modelFile);


%% Helper function 1: Extract normalized pixel features

function features = imagesToFeatures(imds, imageHeight, imageWidth)

    numberOfImages = numel(imds.Files);
    numberOfPixels = imageHeight * imageWidth;

    features = zeros( ...
        numberOfImages, numberOfPixels, "single");

    for imageIndex = 1:numberOfImages

        image = readimage(imds, imageIndex);

        assert(isequal(size(image), ...
            [imageHeight, imageWidth]), ...
            "Unexpected image size in: %s", ...
            imds.Files{imageIndex});

        image = im2single(image);

        features(imageIndex, :) = reshape(image, 1, []);
    end
end


%% Helper function 2: Train the MLP using backpropagation

function [model, validationHistory, bestEpoch] = trainMLP( ...
    X, Y, hiddenSize, maxEpochs, batchSize, ...
    learningRate, lambda, seed, XValidation, YValidation, patience)

    rng(seed);

    numberOfSamples = size(X, 1);
    inputSize = size(X, 2);

    % Xavier-style initialization.
    W1 = single(randn(inputSize, hiddenSize) * ...
        sqrt(2 / (inputSize + hiddenSize)));

    b1 = zeros(1, hiddenSize, "single");

    W2 = single(randn(hiddenSize, 1) * ...
        sqrt(2 / (hiddenSize + 1)));

    b2 = zeros(1, 1, "single");

    validationHistory = [];

    bestEpoch = maxEpochs;

    bestValidationLoss = inf;
    epochsWithoutImprovement = 0;

    bestW1 = W1;
    bestb1 = b1;
    bestW2 = W2;
    bestb2 = b2;

    useValidation = ~isempty(XValidation);

    for epoch = 1:maxEpochs

        % Shuffle the training samples.
        shuffledIndices = randperm(numberOfSamples);

        for batchStart = 1:batchSize:numberOfSamples

            batchEnd = min( ...
                batchStart + batchSize - 1, numberOfSamples);

            batchIndices = shuffledIndices(batchStart:batchEnd);

            XBatch = X(batchIndices, :);
            YBatch = Y(batchIndices, :);

            batchCount = size(XBatch, 1);

            %% Forward propagation

            Z1 = XBatch * W1 + b1;

            A1 = sigmoidMLP(Z1);

            Z2 = A1 * W2 + b2;

            A2 = sigmoidMLP(Z2);

            %% Backpropagation

            % Output-layer error for sigmoid + binary cross-entropy.
            delta2 = A2 - YBatch;

            % Output-layer gradients.
            dW2 = (A1' * delta2) / batchCount ...
                + lambda * W2;

            db2 = mean(delta2, 1);

            % Hidden-layer error.
            delta1 = (delta2 * W2') .* A1 .* (1 - A1);

            % Hidden-layer gradients.
            dW1 = (XBatch' * delta1) / batchCount ...
                + lambda * W1;

            db1 = mean(delta1, 1);

            %% Update weights and biases

            W1 = W1 - learningRate * dW1;
            b1 = b1 - learningRate * db1;

            W2 = W2 - learningRate * dW2;
            b2 = b2 - learningRate * db2;

        end

        %% Validation and early stopping

        if useValidation

            validationScores = predictMLP( ...
                struct("W1", W1, "b1", b1, ...
                       "W2", W2, "b2", b2), ...
                XValidation);

            validationLoss = binaryCrossEntropy( ...
                YValidation, validationScores);

            validationHistory(end + 1, 1) = ...
                validationLoss;

            if validationLoss < bestValidationLoss - 1e-6

                bestValidationLoss = validationLoss;

                bestEpoch = epoch;

                bestW1 = W1;
                bestb1 = b1;
                bestW2 = W2;
                bestb2 = b2;

                epochsWithoutImprovement = 0;

            else

                epochsWithoutImprovement = ...
                    epochsWithoutImprovement + 1;

            end

            if epochsWithoutImprovement >= patience
                fprintf("Early stopping at epoch %d.\n", epoch);
                break;
            end

        end

    end

    if useValidation

        % Return the best validation checkpoint.
        W1 = bestW1;
        b1 = bestb1;
        W2 = bestW2;
        b2 = bestb2;

    end

    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.b2 = b2;
end


%% Helper function 3: Predict class scores

function scores = predictMLP(model, X)

    A1 = sigmoidMLP(X * model.W1 + model.b1);

    scores = sigmoidMLP(A1 * model.W2 + model.b2);
end


%% Helper function 4: Sigmoid activation

function y = sigmoidMLP(x)

    y = 1 ./ (1 + exp(-x));
end


%% Helper function 5: Binary cross-entropy loss

function loss = binaryCrossEntropy(Y, scores)

    scores = min(max(scores, 1e-7), 1 - 1e-7);

    loss = -mean( ...
        Y .* log(scores) + ...
        (1 - Y) .* log(1 - scores));

    loss = double(loss);
end
