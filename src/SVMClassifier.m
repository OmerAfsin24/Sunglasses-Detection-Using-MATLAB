%% SVMClassifier.m
% Train and evaluate a linear SVM for sunglasses detection.

clear; clc; close all;

rng(42);

%% 1. Define project paths

% Directory containing this script (src).
srcRoot = fileparts(mfilename("fullpath"));

% Root directory of the project.
projectRoot = fileparts(srcRoot);

% Prepared dataset directory.
datasetRoot = fullfile(projectRoot, "dataset");

% Verify that the dataset exists.
assert(isfolder(datasetRoot), ...
    "Dataset directory not found: %s. Run prepare_dataset.m first.", ...
    datasetRoot);

% Directory for trained models.
modelsRoot = fullfile(projectRoot, "models");

% Create the models directory if it does not exist.
if ~isfolder(modelsRoot)
    mkdir(modelsRoot);
end


% Path to the saved SVM model.
modelFile = fullfile(modelsRoot, "trainedSVMModel.mat");

% Directory for saving result visualizations.
imagesRoot = fullfile(projectRoot, "images");

% Create the images directory if it does not exist.
if ~isfolder(imagesRoot)
    mkdir(imagesRoot);
end


%% 2. Read the prepared dataset

rng(42);  % Makes the randomized split reproducible

imds = imageDatastore(datasetRoot, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% Confirm that the dataset contains exactly two classes.
assert(numel(categories(imds.Labels)) == 2, ...
    "The dataset must contain exactly two class folders.");

disp("Full dataset:");
disp(countEachLabel(imds));

%% 3. Create the held-out test set

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.80, "randomized");

disp("Training data:");
disp(countEachLabel(imdsTrain));

disp("Test data:");
disp(countEachLabel(imdsTest));

%% 4. Define 5-fold cross-validation

cvPartition = cvpartition(imdsTrain.Labels, "KFold", 5);

%% 5. Convert images into normalized feature vectors

imageHeight = 120;
imageWidth = 128;

XTrain = imagesToFeatures(imdsTrain, imageHeight, imageWidth);
YTrain = imdsTrain.Labels;

XTest = imagesToFeatures(imdsTest, imageHeight, imageWidth);
YTest = imdsTest.Labels;

%% 6. Tune BoxConstraint with 5-fold cross-validation

numberOfFolds = cvPartition.NumTestSets;

% Candidate C values for the linear SVM.
boxConstraintValues = [0.001, 0.01, 0.1, 1, 10, 100];

% Rows represent C values; columns represent CV folds.
validationAccuracies = zeros( ...
    numel(boxConstraintValues), numberOfFolds);

for parameterIndex = 1:numel(boxConstraintValues)

    boxConstraint = boxConstraintValues(parameterIndex);

    for fold = 1:numberOfFolds

        % Obtain the training and validation rows for this fold.
        foldTrainingIndices = training(cvPartition, fold);
        foldValidationIndices = test(cvPartition, fold);

        XFoldTrain = XTrain(foldTrainingIndices, :);
        YFoldTrain = YTrain(foldTrainingIndices);

        XValidation = XTrain(foldValidationIndices, :);
        YValidation = YTrain(foldValidationIndices);

        % Train a linear SVM using the current C value.
        svmModel = fitcsvm(XFoldTrain, YFoldTrain, ...
            KernelFunction="linear", ...
            BoxConstraint=boxConstraint, ...
            Standardize=false);

        % Evaluate on the validation subset for this fold.
        predictedLabels = predict(svmModel, XValidation);

        validationAccuracies(parameterIndex, fold) = ...
            mean(predictedLabels == YValidation);
    end
end

% Average each candidate C value over all five folds.
meanValidationAccuracies = mean(validationAccuracies, 2);

% Select the C value with the highest mean validation accuracy.
[bestValidationAccuracy, bestIndex] = ...
    max(meanValidationAccuracies);

bestBoxConstraint = boxConstraintValues(bestIndex);

gridSearchResults = table( ...
    boxConstraintValues', ...
    meanValidationAccuracies, ...
    VariableNames=["BoxConstraint", "MeanValidationAccuracy"]);

disp(gridSearchResults);

fprintf("Best BoxConstraint: %g\n", bestBoxConstraint);

fprintf("Best mean CV accuracy: %.2f%%\n", ...
    100 * bestValidationAccuracy);

% Visualize and Save cross-validation results
figure("Name", "Cross-Validation Accuracy");

semilogx(boxConstraintValues, ...
    meanValidationAccuracies * 100, ...
    "-o", ...
    LineWidth=2, ...
    MarkerSize=8);

grid on;

xlabel("BoxConstraint (C)");
ylabel("Mean Validation Accuracy (%)");

title("SVM Hyperparameter Tuning");

exportgraphics(gcf, ...
    fullfile(imagesRoot, "SVM_cross_validation_accuracy.png"), ...
    Resolution=300);


%% 7. Train the final model using the selected hyperparameter

finalSvmModel = fitcsvm(XTrain, YTrain, ...
    KernelFunction="linear", ...
    BoxConstraint=bestBoxConstraint, ...
    Standardize=false);

%% 8. Save the trained SVM model

save(modelFile, ...
    "finalSvmModel", ...
    "bestBoxConstraint", ...
    "imageHeight", ...
    "imageWidth");

fprintf("\nSVM model saved to:\n%s\n", modelFile);

%% 9. Evaluate the final model on the held-out test set

%loadedData = load("./models/trainedSVMModel.mat");
%finalSvmModel = loadedData.finalSvmModel;

testPredictedLabels = predict(finalSvmModel, XTest);

testAccuracy = mean(testPredictedLabels == YTest);

fprintf("Held-out test accuracy: %.2f%%\n", ...
    100 * testAccuracy);

figure("Name", "Test-Set Confusion Matrix");

confusionchart(YTest, testPredictedLabels);


exportgraphics(gcf, ...
    fullfile(imagesRoot, "SVM_confusion_matrix.png"), ...
    Resolution=300);
title("Linear SVM: Sunglasses Detection");


%% 10. Calculate additional evaluation metrics

positiveClassName = "sunglasses";

classNames = categories(YTest);

positiveClassIndex = find( ...
    string(classNames) == positiveClassName, 1);

assert(~isempty(positiveClassIndex), ...
    "Positive class not found in YTest categories.");

negativeClassIndex = setdiff( ...
    1:numel(classNames), positiveClassIndex);

assert(numel(negativeClassIndex) == 1, ...
    "Expected exactly two classes.");

% Create confusion matrix with an explicit class order.
confusionMatrix = confusionmat(YTest, testPredictedLabels, ...
    Order=categorical(classNames, classNames));

% Rows represent true labels; columns represent predicted labels.
TP = confusionMatrix(positiveClassIndex, positiveClassIndex);

FN = sum(confusionMatrix(positiveClassIndex, :)) - TP;

FP = sum(confusionMatrix(:, positiveClassIndex)) - TP;

TN = confusionMatrix(negativeClassIndex, negativeClassIndex);

% Calculate performance metrics.
accuracy = (TP + TN) / sum(confusionMatrix, "all");

precision = TP / max(TP + FP, 1);

recall = TP / max(TP + FN, 1);

specificity = TN / max(TN + FP, 1);

f1Score = 2 * precision * recall / ...
    max(precision + recall, eps);

balancedAccuracy = (recall + specificity) / 2;

fprintf("\nAdditional Test Metrics:\n");

fprintf("Accuracy:          %.4f\n", accuracy);
fprintf("Precision:         %.4f\n", precision);
fprintf("Recall:            %.4f\n", recall);
fprintf("Specificity:       %.4f\n", specificity);
fprintf("F1-score:          %.4f\n", f1Score);
fprintf("Balanced accuracy: %.4f\n", balancedAccuracy);

%% 11. Display random test images and their predictions

numberOfImagesToShow = min(10, numel(imdsTest.Files));

testImageIndices = randperm( ...
    numel(imdsTest.Files), numberOfImagesToShow);

numberOfColumns = 5;

numberOfRows = ceil(numberOfImagesToShow / numberOfColumns);

figure("Name", "Random Test Images and Predictions");

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
        titleColor = [0, 0.5, 0];  % Green: correct prediction
    else
        titleColor = [0.8, 0, 0];  % Red: incorrect prediction
    end

    title(sprintf("True: %s\nPredicted: %s", ...
        string(trueLabel), string(predictedLabel)), ...
        Color=titleColor, ...
        Interpreter="none");
end

sgtitle("Random Held-Out Test Images: Green = Correct, Red = Incorrect");

exportgraphics(gcf, ...
    fullfile(imagesRoot, "SVM_test_predictions.png"), ...
    Resolution=300);


%% 12. Display misclassified test images

misclassifiedIndices = find( ...
    testPredictedLabels ~= YTest);

fprintf("Misclassified images: %d / %d\n", ...
    numel(misclassifiedIndices), numel(YTest));

if ~isempty(misclassifiedIndices)

    numberOfErrorsToShow = min(12, ...
        numel(misclassifiedIndices));

    figure("Name", "Misclassified Test Images");

    tiledlayout( ...
        ceil(numberOfErrorsToShow / 4), 4, ...
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

    sgtitle("Misclassified Test Images");
    exportgraphics(gcf, ...
    fullfile(imagesRoot, "SVM_misclassified_images.png"), ...
    Resolution=300);

end

%% Local function: Convert images into feature vectors

function features = imagesToFeatures(imds, imageHeight, imageWidth)

    numberOfImages = numel(imds.Files);

    numberOfPixels = imageHeight * imageWidth;

    features = zeros(numberOfImages, numberOfPixels, "single");

    for imageIndex = 1:numberOfImages

        image = readimage(imds, imageIndex);

        % Verify that each original CMU image has the expected size.
        assert(isequal(size(image), [imageHeight, imageWidth]), ...
            "Unexpected image size in: %s", ...
            imds.Files{imageIndex});

        % Convert pixel values from [0, 255] to single values in [0, 1].
        image = im2single(image);

        % Convert the 2-D image into one feature row.
        features(imageIndex, :) = reshape(image, 1, []);

    end
end