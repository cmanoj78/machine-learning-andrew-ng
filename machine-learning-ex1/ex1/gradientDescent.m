function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
delta = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % alpha * (1/m) * summation(1-m) [h(xi) - yi]*x 
    %disp('display theta ');
    %disp(theta);
    delta = (alpha/m)*X'*((X * theta)-y);
    theta = theta - delta
    %disp('size of theta')
    %disp(size(theta));
    %disp('iteration = ')
    %disp(iter)
    %disp('size of X');
    %disp(size((X(iter,:))'))
    %theta = theta - (delta*X(iter,:))';
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
