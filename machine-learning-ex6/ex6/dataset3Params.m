function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Setting trial values for C and sigma.
Ctrial = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmatrial = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
errors = ones(size(Ctrial,1),size(sigmatrial,1));

% find the predictions and error for each trial C and sigma combination.
for i=1:size(Ctrial,1)
    for j=1:size(sigmatrial,1)
        %model= svmTrain(X, y, Ctrial(i), @(x,X) gaussianKernel(X, X, sigmatrial(j)));
        model = svmTrain(X, y, Ctrial(i), @(a, b) gaussianKernel(a, b, sigmatrial(j)));
        predictions = svmPredict(model, Xval);
        errors(i,j) = mean(double(predictions ~= yval));
    end
end


maxAccuracy = min(errors(:));
[i,j]= find(errors==maxAccuracy);

C = Ctrial(i,1);
sigma = sigmatrial(j,1);

% =========================================================================

end
