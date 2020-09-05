% Function to compute cost and gradient 
function [J, grad] = CH5019_GRP28_Costfunction(theta, X, y)

m = length(y);

J = 0;
grad = zeros(size(theta));

h = CH5019_GRP28_sigmoid(X*theta);
J = -(1/m)*sum(y'*log(h)+(1-y)'*log(1-h));
grad = (X'*(h-y))./m;
    
end
