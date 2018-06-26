function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%X=[ones(m,1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X=[ones(m,1) X];
h=(X*theta);
J=1/(2*m)*sum((h-y).^2)+(lambda/(2*m))*(sum(theta.^2)-theta(1).^2);
p=(h-y)'*X;
grad=(1/m.*((h-y)'*X))'+(lambda/m).*theta;
grad(1)=1/m*p(1);









% =========================================================================

grad = grad(:);

end
