function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

fprintf( '---------begin' );

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    h = X * theta;
    vec_err = h - y;

    
    %for it = 1 : m
    %  fprintf( '  %f  ', vec_err(it) );
    %end
    %  fprintf( '\n\n' );
    
    %change = vec_err' * X;
    change = X' * vec_err;
    
    unit_change = change/m;
    theta = theta - alpha * unit_change;
    
    %fprintf( '\n unit_change %f %f', unit_change(1), unit_change(2));
    %fprintf( '\n theta %f %f', theta(1), theta(2) );

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    %fprintf( '\n\t\t  %f  ', theta );
end

%fprintf( '---------------end %f %f', theta );

end
