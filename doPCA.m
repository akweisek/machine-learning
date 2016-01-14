%Principal Component Analysis
%Samuel Akwei-Sekyere , 2015

%Input : Vector=Vector (let each ROW be a new dimension), 
%        dimType=representation of number of dimension (as percentage of
%                   variance or explicitly as an integer)
%        dimType=number of dimensions in resulting pca (percent or integer)
%
%Output: PC=matrix with results in their right dimensions in each ROW

function [PC,S,variances,V]=doPCA(vector,dimType,dimNo,plotScree)

    %Get covariance matrix
    [Sigma,normalizedVector] = covarianceMatrix(vector);
    
    %Eigendecomposition
    [U,S,V]=svd(Sigma);
    
    %number of eigenvectors needed
    switch dimType
        case 'integer'
            U=U(:,1:dimNo); %colums for eigenvectors
        case 'percentage'
            percentages = cumsum(diag(S)); %cumulative summation of the diagonal matrix S
            [~,dimension] = min(abs(percentages - dimNo)); %dimension closest to the percentage of variance
            U=U(:,1:dimension);
    end
    
    %Eigenvector projection
    PC=U'*normalizedVector';

    variances=diag(S).*diag(S);

    if plotScree
        figure;
        plot(variances,'o-')
        xlabel('eigenvector number');
        ylabel('eigenvalue');
        title('Eigenspectrum');
    end

    %Covariance matrix
    function [Sigma,Vector] = covarianceMatrix(Vector)
        Vector=Vector';
        nSamples = length(Vector(:,1));

        onesVector = ones(nSamples,1);
        Vector = Vector - onesVector*onesVector'*Vector*(1/nSamples); %get deviation scores

        Sigma = Vector'*Vector*(1/nSamples);