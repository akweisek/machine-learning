%Samuel Akwei-Sekyere
%Least Square regression
%
%INPUTS:
%       mode -> either 'learn' or 'test'
%
%       inputData -> vector as DataPoints x DIMENSION (each column is a new
%                               dimension) either learning or testing set
%
%       outputData_coefficients -> column vector of data output (if 'learn');
%                                  column vector of polynomial coefficients (if 'test')
%
%       polynomialDegree -> the polynomial degree to be considered

function [coefficients,RMSE,testingOutput] = leastSquareRegressionPolynomial(mode, inputData, outputData_coefficients , polynomialDegree)

switch mode
    case 'learn'
        [coefficients,RMSE] = least_squares (inputData,outputData_coefficients,polynomialDegree);
        testingOutput = [];
        
    case 'test'
        [testingOutput]=least_squares_proj(outputData_coefficients,inputData,polynomialDegree);
        coefficients = outputData_coefficients;
        RMSE = [];
end

%--------------------------------------------------------------------------------------------------
%Learn via least squares
%
%Finds a,b,c for y = ax^n + ... + bx + ... c
%
%Assume you have some data {(x1,y1),...,(xn,yn)}
%Let each row of G be a new sample point
% (and each column be a new feature/variable/axis;
        %in this case you will be finding a,b,c for 
        %z = ax^n + by^n + ... + cx + dy + e)
   %
   %AND Let y be a column matrix
% Output === this is the a,b,c
function [coefficients,RMSE] = least_squares (G,y,order)

var=length(G(1,:)); %number of columns gives me how many variables/features
exponents=fliplr(1:order); %getting the exponents thus x^n + x + whatever
A=ones(length(G(:,1)),var.*length(exponents)+1); %each row is a sample point and each column is the order

m = size(G); m = m(1)*m(2);

    for k=exponents;
        index=order-k+1;
        exp_G=G.^k;
      %  col_index=sum(exp_G,2); %sum each column/variable/axis
        A(:,index*var-(var-1):index*var)=exp_G;
    end
    
    coefficients=pinv(A'*A)*A'*y; %least squares solution
    RMSE = sqrt(sum((y - A*coefficients).^2))/sqrt(m);

%----------------------------------------------------------------------------------------------


%Inputs:
%       theta -- this is a single column which you obtained by calling
%       least_squares.m
%
%       Let each row of G be a new sample point
%       (and each column be a new feature/variable/axis;
%
%       order ---- the order of the polynomial
%
%Outputs:
%       out ---- returns z for z=ax^n + by^n + ... + dx + ey + f

function [out]=least_squares_proj(theta,G,order)

    var=length(G(1,:));

    exponents = order:-1:0;
    
    expData = length(exponents);
    
    lengthTheta = length(theta);
    
    lengthData = length(G(:,1));
    
    out = zeros(lengthData,1);
    
    exponentCoefficient = [];
    for i = 1:expData-1
        expCurrent = repmat(exponents(i),[1 var]);
        exponentCoefficient = horzcat(exponentCoefficient,expCurrent);
    end
    exponentCoefficient = horzcat(exponentCoefficient,0); %not necessary!
    
    
    for i = 1:lengthData
        currentInput = G(i,:);
        currentOutput = 0;
        for j = 1:lengthTheta-1
            currentIndexDimension = mod((j-1),var)+1;
            
            currentOutput = currentOutput + ...
                + theta(j)*((currentInput(currentIndexDimension))...
                .^exponentCoefficient(j));
        end
        out(i) = currentOutput + theta(end);
    end
  