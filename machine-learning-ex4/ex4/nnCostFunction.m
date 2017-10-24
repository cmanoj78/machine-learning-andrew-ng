function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%---------------- Change START ----------------------
%   HOW TO compute h_theta 
%           a_1 = x
%           a_2 = g(z_2) = g(Theta1 * x)
%           a_3 = g(z_3) = g(Theta2 * a_2) = h_theta
%           h_theta = g(Theta2 * g(Theta1 * x))
%
%   Note: X is 5000 * 400. We need to add one column of 1 for bias unit
    a_1 = [ones(m,1) X];  % X should now be 5000*401
    a_2 = sigmoid(a_1 * (Theta1)');  % result will be 5000*25 
% add one column of 1 for bias unit.
    a_2 = [ones(size(a_2,1),1) a_2]; % a_2 will now be 5000*26
%           h_theta = a_3 = g(z_3) = g(theta_2 * a_2)
    a_3 = sigmoid(a_2 * (Theta2)'); % result h_theta should be 5000 * 10
    h_theta = a_3;
    y1 = zeros(size(y,1), num_labels); % y1 will be 5000*10
    
    for index = 1:m
        y1(index, y(index)) = 1;
    end
    
    % created debug_var to separate computation and debug.
    debug_var = ((y1 .* log(h_theta)) + ((1-y1) .* log(1-h_theta)));

    J = -1/m * sum(sum(debug_var)) ;
% IMPORTANT : pls note that the product (in debug_var) is a element wise
% product. I made a mistake of doing matrix product to get 10x10 result
% which was incorrect. If you look at the formula y and h_theta elements 
% have no cooreation except that they are same size ie a matrix product
% will happen as diemensions coincide but it is meaningless. you have to
% carefully look if the product really makes sense esp when doing matrix
% product.
    
% adding regularizatoin term = lambda/2m * sum all theta square    

penalty = ( lambda / (2*m) ) * (sum(sum((Theta1(:,2:end).^2)))+ ...
    sum(sum(Theta2(:,2:end).^2)));

J = J + penalty;
    %---------------- Change END -----------------e-----





% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%---------------- Change START ----------------------
% the exercize asks to compute this using for loop. Adding '_bp' to all
% variable names to refer individual sample for each data set.

d_3_bp = zeros(size(y1,2),1); % no _bp here. d_3 does not change by data set

Delta_2 = zeros(size(Theta2)); % Theta2 is 10*26
Delta_1 = zeros(size(Theta1)); % Theta1 is 25*401
for t=1:m
    % compute a_1 as 1 x
    a_1_bp = [1 ; X(t,:)']; %a_1 should be 401*1
    % a_1_bp = a_1(t :)'
    z_2_bp = Theta1 * a_1_bp;  % (25*401)*(401*1) = result 25*1
    a_2_bp = [1; sigmoid(z_2_bp)];  % after adding ones 26*1
    z_3_bp = Theta2 * a_2_bp; % (10*26) * (26*1) = result (10*1)
    a_3_bp = sigmoid(z_3_bp);  % no adding ones here as this is output layer. 10*1
    d_3_bp = a_3_bp - y1(t,:)'; % d_3 will be 10*1
    d_2_bp = Theta2' * d_3_bp .* a_2_bp .* (1-a_2_bp); % (10*26)'*(10*1) .*(26*1).*(26*1)
    d_2_bp = d_2_bp(2:end); % removing d_2[0]; resi;t is 25*1
    Delta_2 = Delta_2 + (d_3_bp * a_2_bp'); % prod = (10*1)*(26*1)' = 10*26
    Delta_1 = Delta_1 + (d_2_bp * a_1_bp'); % prod = (25*1) * (401*1)' = 25*401
end % for

    Theta1_grad = (1/m)*Delta_1; % 25*401
    Theta2_grad = (1/m)*Delta_2; % 10*26


%---------------- Change END ----------------------

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%---------------- Change START ----------------------

penalty_1 = zeros(size(Theta1,1),size(Theta1,2)-1); % 25*400
penalty_2 = zeros(size(Theta2,1),size(Theta2,2)-1); % 10*25

penalty_1 = (lambda / m)* Theta1(:,2:end);
penalty_2 = (lambda / m)* Theta2(:,2:end);

Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end) + penalty_1];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end) + penalty_2];

%---------------- Change END ----------------------

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
