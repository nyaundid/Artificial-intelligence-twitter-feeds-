opts = detectImportOptions('C:\Users\Meshal\Desktop\Fall 2018\Artificial Intelligence\Project\DS\twcs.csv');
T = readtable('C:\Users\Meshal\Desktop\Fall 2018\Artificial Intelligence\Project\DS\twcs.csv',opts);


%classify-text-data-using-deep-learning
%The goal of this example is to classify events by the label in the event_type column. To divide the data into classes, convert these labels to categorical.
T.inbound = categorical(T.inbound);


%split
cvp = cvpartition(T.inbound,'Holdout',0.3);
dataTrain = T(training(cvp),:);
dataHeldOut = T(test(cvp),:);

cvp = cvpartition(dataHeldOut.inbound,'HoldOut',0.5);
dataValidation = dataHeldOut(training(cvp),:);
dataTest = dataHeldOut(test(cvp),:);

%extract the target
YTrain = dataTrain.inbound;
YValidation = dataValidation.inbound;
YTest = dataTest.inbound;

%Preprocess Training Text Data
%remove twitter user names
textDataTrain = regexprep(dataTrain.text, '@[a-zA-Z0-9_]{0,15}', ' ', 'all');

%remove non-english characters and punctuations
textDataTrain = regexprep(textDataTrain, '[^A-Za-z]', ' ', 'all');



textDataTrain = lower(textDataTrain);
documentsTrain = tokenizedDocument(textDataTrain);

documentsTrain = removeStopWords(documentsTrain);

%Remove words with 2 or fewer characters, and words with 15 or greater characters.

documentsTrain = removeShortWords(documentsTrain,2);
documentsTrain = removeLongWords(documentsTrain,15);

%Remove Empty Documents
%documentsTrain = removeEmptyDocuments(documentsTrain);
[documentsTrain,idx] = removeEmptyDocuments(documentsTrain);
%Removing empty documents targets
YTrain(idx) = [];

%Lemmatize the words using normalizeWords. To improve lemmatization, first add part of speech details to the documents using addPartOfSpeechDetails.

documentsTrain = addPartOfSpeechDetails(documentsTrain);
documentsTrain = normalizeWords(documentsTrain,'Style','lemma');

%-----------------------------------------------

%Preprocess Validation Text Data
%remove twitter user names
textDataValidation = regexprep(dataValidation.text, '@[a-zA-Z0-9_]{0,15}', ' ', 'all');

%remove non-english characters and punctuations
textDataValidation = regexprep(textDataValidation, '[^A-Za-z]', ' ', 'all');



textDataValidation = lower(textDataValidation);
documentsValidation = tokenizedDocument(textDataValidation);

documentsValidation = removeStopWords(documentsValidation);

%Remove words with 2 or fewer characters, and words with 15 or greater characters.

documentsValidation = removeShortWords(documentsValidation,2);
documentsValidation = removeLongWords(documentsValidation,15);

%Remove Empty Documents
%documentsValidation = removeEmptyDocuments(documentsValidation);
[documentsValidation,idx1] = removeEmptyDocuments(documentsValidation);
%Removing empty documents targets
YValidation(idx1) = [];

%Lemmatize the words using normalizeWords. To improve lemmatization, first add part of speech details to the documents using addPartOfSpeechDetails.

documentsValidation = addPartOfSpeechDetails(documentsValidation);
documentsValidation = normalizeWords(documentsValidation,'Style','lemma');

%-----------------------------------------------

%Preprocess Test Text Data
%remove twitter user names
textDataTest = regexprep(dataTest.text, '@[a-zA-Z0-9_]{0,15}', ' ', 'all');

%remove non-english characters and punctuations
textDataTest = regexprep(textDataTest, '[^A-Za-z]', ' ', 'all');



textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);

documentsTest = removeStopWords(documentsTest);

%Remove words with 2 or fewer characters, and words with 15 or greater characters.

documentsTest = removeShortWords(documentsTest,2);
documentsTest = removeLongWords(documentsTest,15);

%Remove Empty Documents
%documentsTest = removeEmptyDocuments(documentsTest);
[documentsTest,idx2] = removeEmptyDocuments(documentsTest);
%Removing empty documents targets
YTest(idx2) = [];

%Lemmatize the words using normalizeWords. To improve lemmatization, first add part of speech details to the documents using addPartOfSpeechDetails.

documentsTest = addPartOfSpeechDetails(documentsTest);
documentsTest = normalizeWords(documentsTest,'Style','lemma');


%Convert Document to Sequences
%To input the documents into an LSTM network, use a word encoding to convert the documents into sequences of numeric indices.
enc = wordEncoding(documentsTrain);


%---------------------------------------------------
%histogram of the training document lengths.
%documentLengths = doclength(documentsTrain);
%figure
%histogram(documentLengths)
%title("Document Lengths")
%xlabel("Length")
%ylabel("Number of Documents")
%---------------------------------------------------

%Convert the documents to sequences of numeric indices using doc2sequence. To truncate or left-pad the sequences to have length 15, set the 'Length' option to 15.
XTrain = doc2sequence(enc,documentsTrain,'Length',13);
%XTrain(1:5)

%Convert the documents to sequences Validation data set
XValidation = doc2sequence(enc,documentsValidation,'Length',13);

%Create and Train LSTM Network
inputSize = 1;
embeddingDimension = 100;
numHiddenUnits = enc.NumWords;
hiddenSize = 180;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numHiddenUnits)
    lstmLayer(hiddenSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);

%Test LSTM Network
%XTest = doc2sequence(enc,documentsTest,'Length',13);
%XTest(1:5)

Y1 = dummyvar(YTest(:, 1))';
Y2 = dummyvar(YPred(:, 1))';

%YPred = classify(net,XTest);
%accuracy = sum(YPred == YTest)/numel(YPred);

plotconfusion(YTest,YPred);
plotconfusion(Y1,Y2);
[c,cm,ind,per] = confusion(Y1,Y2);
TP = cm(2,2);
FP = cm(1,2);
FN = cm(2,1);
TP1 = cm(1,1);
FP1 = cm(2,1);
FN1 = cm(1,2);
plotroc(Y1,Y2);
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
Precision1 = TP1 / (TP1 + FP1);
Recall1 = TP1 / (TP1 + FN1);
F1 = 2 * (Precision * Recall) / ((1/Precision) + (1/Recall));
F11 = 2 * (Precision1 * Recall1) / ((1/Precision1) + (1/Recall1));
%analyzeNetwork(net)


