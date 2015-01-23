function [cvMCR,cvConfusion] = cvclassify(X,y,cvNumFolds, DoConfusion)
%% Preprocess Data
%load('fisheriris.mat')
%y = double(nominal(species));
%X = meas;
%X = meas(:,[1 3 4]);
X = double(X);
y = double(y);

%% Choose classifier
%UseLogisticRegression = false;
UseLogisticRegression = true;
if UseLogisticRegression
    % Multinomial Logistic Regression
    fcn_classify = @classify_mnlr;
    fprintf('Using Logistic Regression Classifier...\n');
else   
    % Discriminant Analysis
    fprintf('Using Built-in (Linear) Discriminant Classifier...\n');
    fcn_classify = @classify_discr;
end


%% Cross Validation
cvp = cvpartition(y,'kfold',cvNumFolds) % Stratified cross-validation

% CV of Misclassification Rate (MCR)
cvMCR = crossval('mcr',X,y,'predfun',fcn_classify,'partition',cvp);

% CV of Confusion
cvConfusion = [];
if DoConfusion
    order = unique(y); % Order of the group labels
    fcn_confusion = @(Xtrain,ytrain,Xtest,ytest)...
        confusionmat(ytest, fcn_classify(Xtrain,ytrain,Xtest), 'order', order);

    cvConfusion = crossval(fcn_confusion,X,y,'partition',cvp);
    cvConfusion = reshape(sum(cvConfusion),length(order),length(order));
end

fprintf('Done.\n');

end

function [yhat] = classify_discr(Xtrain, ytrain, Xtest)
    % IMPORTANT: classify takes args in DIFFERENT ORDER than
    % that expected in crossval usage!
    [yhat] = classify(Xtest, Xtrain, ytrain);
end

function [yhat] = classify_mnlr(Xtrain, ytrain, Xtest)
    % IMPORTANT: ytrain might be nominal, must cast to double
    B = mnrfit(Xtrain, double(ytrain));
    pihat = mnrval(B,Xtest);
    [~,yhat] = max(pihat,[],2);
end

function [pihat] = post_probs_mnlr(Xtrain, ytrain, Xtest)
    % IMPORTANT: ytrain might be nominal, must cast to double
    B = mnrfit(Xtrain, double(ytrain));
    pihat = mnrval(B,Xtest);
    %[~,yhat] = max(pihat,[],2);
end
