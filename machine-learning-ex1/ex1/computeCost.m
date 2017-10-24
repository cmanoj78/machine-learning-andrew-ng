function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

%squareValue = 0;
%m=length(y)
%for i=1:m,
%    temp = (theta(1)*X(i,1))+(theta(2)*X(i,2))-y(i);
%    squareValue = squareValue + temp^2;
%end
%J = (0.5/m)*squareValue;
disp('computing new J')
J=(0.5/m)*(sum(((X*theta)-y).^2))

end
