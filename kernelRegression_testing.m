%Samuel Akwei-Sekyere, 2015
%
%
%INPUTS:
%           testingInput -> testing set (input) let each row be a new
%                           dimension
%
%           K -> the kernel convolution K obtained from
%                kernelRegression_training
%
%           c -> the coefficients obtained from kernelRegression_training

function y = kernelRegression_testing(testingInput, trainingSet_input , c, params)

    n = length(trainingSet_input(1,:));
    p = length(testingInput(1,:));
    y = zeros(length(c(1,:)),p);
    sigma = params;
    for z = 1:p
        K = zeros(1,n);
        
        for i = 1:n
            currentSet_i = trainingSet_input(:,i);
            inputCheck_z = testingInput(:,z);
            
            euclideanDistance = sum((currentSet_i - inputCheck_z).^2);
            
            K(1,i) = exp(-euclideanDistance/(2.*(sigma.^2)));
        end
        
        y(z) = K*c;
    end