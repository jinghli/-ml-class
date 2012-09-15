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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%J
a1 = X;
a1_bias = [ones(m,1) a1];

z2 = a1_bias * Theta1';
a2 = sigmoid(z2);
a2_bias = [ones(size(a2, 1),1) a2];

z3 = a2_bias * Theta2';
a3 = sigmoid(z3);
h = a3;


c = 0;
for k=1:num_labels
  c_local = -(y==k)' * log(h(:,k)) - (1-(y==k)') * log(1-h(:,k));
  c += c_local;
end;
J = 1/m*sum(c);

%Regularized.
%start from row 2.
theta1_no_bias = Theta1(:,2:end);
theta2_no_bias = Theta2(:,2:end);

reg = lambda/2/m * (sum(sum(theta1_no_bias.^2)) + sum(sum(theta2_no_bias.^2)));
J = J + reg;


%response position is 1
%5000 * 10
yt = [];
for k=1:num_labels
  yt = [yt y==k];
end;

big_delta_2 = zeros(size(Theta2));
big_delta_1 = zeros(size(Theta1));

for t=1:m
  %delta 3
  delta_3 = a3(t,:) - yt(t,:);
  %delta 2
  delta_2 = delta_3 * theta2_no_bias .* sigmoidGradient(z2(t,:));
  
  %big_delta 2 and 1
  big_delta_2 += delta_3' * a2_bias(t,:);
  big_delta_1 += delta_2' * a1_bias(t,:);

end;

Theta1_l = Theta1;
%change the 1st column = 0
%i start from 1, but j start from 0
Theta1_l(:,1) = 0;
Theta2_l = Theta2;
%change the 1st column = 0
Theta2_l(:,1) = 0;
Theta2_grad = (big_delta_2 + lambda * Theta2_l) / m;
Theta1_grad = (big_delta_1 + lambda * Theta1_l) / m;




% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
