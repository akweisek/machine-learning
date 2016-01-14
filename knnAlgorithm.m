%Samuel Akwei-Sekyere, 2015
%K nearest neigbor
%
%Inputs: 
%    ----TRAINING SET------------
%        dictionaryInput -> dictionary input (to look-up) (vectors)
%                               Note: each new ROW is a new dimension
%        dictionaryOutput -> dictionary output (results) (vector)
%
%    ----LEARNING RULE-----------
%        distanceMetric -> the distance metric to be used ('string')
%        distanceFactor -> parameter of distance metrics that need extra
%                           parameters (eg. Minkowski distance)
%
%    ----TESTING------------------
%        searchQuery -> what to look up in knn
%        numNearestNeighbor -> number of nearest neighbors to extract (int)
%
%
%Outputs: lookupOutput -> the output from looking up

function [lookupOutput,lookupInput] = knnAlgorithm(dictionaryInput,dictionaryOutput,...
    distanceMetric, searchQuery, numNearestNeighbor, distanceFactor)

    if nargin < 6
        distanceFactor = 2;
    end

    lenInput = length(dictionaryInput(1,:)); %number input of vectors (cols)
    
    lenSearch = length(searchQuery(1,:)); %number of things to search (cols)
    
    lookupOutput = cell(1,lenSearch); %initializing the lookup outputs
    
    lookupInput = cell(1,lenSearch); %initializing the lookup inputs

    switch distanceMetric
        case 'manhattan'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = sum(abs(dictionaryInput-repSearchData),1);
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest output vectors
            end
            
        case 'meanAbsoluteError'
            numDimension = length(searchQuery(:,1));
            
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = (1./numDimension)...
                    *sum(abs(dictionaryInput-repSearchData),1);
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end 
           
        case 'euclidean'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = sqrt(sum((dictionaryInput-repSearchData).^2,1));
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
         
        case 'sqEuclidean'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = sum((dictionaryInput-repSearchData).^2,1);
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
            
         case 'meanSquaredError'
             numDimension = length(searchQuery(:,1));
             
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = (1./numDimension)...
                    *sum((dictionaryInput-repSearchData).^2,1);
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
            
        case 'Chebyshev'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = max(abs(dictionaryInput-repSearchData));
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
        
        case 'Minkowski'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = (sum((abs(dictionaryInput-repSearchData)).^distanceFactor,1))...
                    .^(1/distanceFactor);
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
            
        case 'Canberra'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                distances = sum((abs(dictionaryInput-repSearchData))...
                   ./(abs(dictionaryInput)+abs(repSearchData)) , 1);
                    
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
            
        case 'cosine'
            for i = 1:lenSearch
                currentSearch = searchQuery(:,i); %current vector (cols)
                
                %repeat current search query matrix
                repSearchData = repmat(currentSearch,[1,lenInput]);
                
                %distance between the current search and all vectors in the dictionary
                dotProduct = sum(dictionaryInput.*repSearchData,1);
                inSqrd = sqrt(sum(dictionaryInput.^2,1));
                outSqrd = sqrt(sum(repSearchData.^2,1));
                prodInOut = inSqrd.*outSqrd;
                
                distances = dotProduct./prodInOut;
                
                %argmin distances to k (sort first)
                [ ~ , index ] = sort(distances);
                
                outputIndices = index(1:numNearestNeighbor);
                
                lookupInput{i} = dictionaryInput(:,outputIndices); %get nearest input vectors
                lookupOutput{i} = dictionaryOutput(:,outputIndices); %get nearest vectors
            end
    end