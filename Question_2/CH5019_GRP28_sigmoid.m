% Function to return the sigmoid value of input
function g = CH5019_GRP28_sigmoid(z)

g = zeros(size(z));
g = 1./(1+exp(-z));
    
end
