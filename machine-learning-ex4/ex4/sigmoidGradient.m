function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

%---------------- Change START ----------------------

g = sigmoid(z).*(1 - sigmoid(z));

% NOTE : the g here is actually g'(z) in notes. This refers to gradient of
% the sigmoid function. the formula is 
% g'(z) = d/dz (g(z)) = g(z) * (1-g(z))
% REMEMBER - we need to elementwise product here. Both g(z) and 1 - g(z)
% will be of the same size.

%---------------- Change END ----------------------


% =============================================================




end
