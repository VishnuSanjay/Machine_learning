function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
p=size(theta);
grad = zeros(p(1),p(2));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=sigmoid(X*theta);
l=-y'*log(h);
r=(1-y)'*log(1-h);
J=((1/m)*(l-r)+(lambda/(2*m))*(sum(theta.^2)-theta(1).^2));
p=(h-y)'*X;
grad=(1/m.*((h-y)'*X))'+(lambda/m).*theta;
grad(1)=1/m*p(1);







% =============================================================

end
