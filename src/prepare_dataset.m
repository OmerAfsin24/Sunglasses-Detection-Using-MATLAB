%% prepare_dataset.m
% Build a binary sunglasses dataset from the CMU Face Images archive.
% Only original images are copied. Numbered variants are ignored.

clear; clc;

%% Project paths

% Find the directory containing this script (src).
srcRoot = fileparts(mfilename("fullpath"));

% The project root is the parent directory of src.
projectRoot = fileparts(srcRoot);

% Raw dataset paths.
rawDataRoot = fullfile(projectRoot, "cmu+face+images");
archivePath = fullfile(rawDataRoot, "faces.tar.gz");
extractionRoot = fullfile(rawDataRoot, "extracted");
facesRoot = fullfile(extractionRoot, "faces");

% Processed dataset paths.
datasetRoot = fullfile(projectRoot, "dataset");
sunglassesRoot = fullfile(datasetRoot, "sunglasses");
noSunglassesRoot = fullfile(datasetRoot, "no_sunglasses");

%% Extract the archive if necessary

% Extract the archive only if the faces directory does not exist.
if ~isfolder(facesRoot)

    assert(isfile(archivePath), ...
        "Archive not found: %s", archivePath);

    if ~isfolder(extractionRoot)
        mkdir(extractionRoot);
    end

    untar(archivePath, extractionRoot);
end

%% Create class-label folders

if ~isfolder(sunglassesRoot)
    mkdir(sunglassesRoot);
end

if ~isfolder(noSunglassesRoot)
    mkdir(noSunglassesRoot);
end

%% Find original PGM images

% Search recursively through all subject directories.
imageFiles = dir(fullfile(facesRoot, "**", "*.pgm"));

fprintf("Total PGM images found: %d\n", numel(imageFiles));

%% Process images

copiedSunglasses = 0;
copiedNoSunglasses = 0;
skippedVariants = 0;

for k = 1:numel(imageFiles)

    fileName = imageFiles(k).name;
    fileNameLower = lower(fileName);

    % Skip numbered variants, e.g.:
    % an2i_left_angry_open_2.pgm
    isNumberedVariant = ~isempty( ...
        regexp(fileNameLower, "_\d+\.pgm$", "once"));

    if isNumberedVariant
        skippedVariants = skippedVariants + 1;
        continue;
    end

    % Source image path.
    sourcePath = fullfile(imageFiles(k).folder, fileName);

    % Assign the image to its class.
    if endsWith(fileNameLower, "_sunglasses.pgm")

        destinationPath = fullfile(sunglassesRoot, fileName);
        copiedSunglasses = copiedSunglasses + 1;

    else

        destinationPath = fullfile(noSunglassesRoot, fileName);
        copiedNoSunglasses = copiedNoSunglasses + 1;

    end

    % Copy the image to its destination.
    copyfile(sourcePath, destinationPath);

end

%% Display results

fprintf("\nDataset preparation completed.\n");
fprintf("Sunglasses images: %d\n", copiedSunglasses);
fprintf("No-sunglasses images: %d\n", copiedNoSunglasses);
fprintf("Skipped numbered variants: %d\n", skippedVariants);
fprintf("Dataset location: %s\n", datasetRoot);
