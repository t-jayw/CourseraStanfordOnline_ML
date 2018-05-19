function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 3;
sigma = .01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%             3  validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%predictions = svmPredict(model, Xval)

foo = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i = 1:length(foo), %C
	for j = 1:length(foo), %sigma
		C = foo(i);
		sigma = foo(j);
		testModel = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

		predictions = svmPredict(testModel, Xval);
		
		testErrors(i,j) = mean(double(predictions ~= yval));
end;
end;

[minColumnError, minColumnErrorIndex] = min(testErrors);
[minError, minErrorIndex] = min(minColumnError);

C = foo(minColumnErrorIndex(minErrorIndex));
sigma = foo(minErrorIndex);


% =========================================================================

end
