function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cc = [0.01,0.03,0.1,0.3,1,3,10,30];
sigmac = [0.01,0.03,0.1,0.3,1,3,10,30];

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
error=realmax;
for i=1:numel(Cc)
    for j=1:numel(sigmac)
        model= svmTrain(X, y, Cc(i), @(x1, x2) gaussianKernel(x1, x2, sigmac(j)));
        ypred=svmPredict(model,Xval);
        error1=mean(double(ypred~=yval));
        if error1<error
            C=Cc(i);
            sigma=sigmac(j);
            error=error1;
        end
    end
end






% =========================================================================

end
