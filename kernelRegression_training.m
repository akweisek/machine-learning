function [c, K] = kernelRegression_training(trainingSet_input,trainingSet_output,kernelType,params)

    switch kernelType
        case 'gaussian'
            sigma = params;
            n = length(trainingSet_input(1,:)); %let each new row be a new dimension
            
            kernelData = zeros(n,n);
            
            for i=1:n
                for j=1:n
                    currentSet_i = trainingSet_input(:,i);
                    currentSet_j = trainingSet_input(:,j);
                    
                    euclideanDistance = sum((currentSet_i - currentSet_j).^2);
                    
                    kernelData(i,j) = exp(-euclideanDistance/(2.*(sigma.^2)));
                end
            end
            
            %Finding least square solution (y = Kc || Kc = y => c = inv(K'*K)*(K'*y) )
            K = kernelData;
            y = trainingSet_output';
            
            c = pinv(K'*K)*(K'*y);
            
    end