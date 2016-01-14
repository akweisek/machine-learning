%Samuel Akwei-Sekyere, 2015
%   learnANN (incomplete)
%
%INPUTS: learningData -> struct containing required elements
%        networkType -> the type of network to implement (eg. feedforward)
%
%   Below is a list of the required data for different network types
%           NOTE : each ROW is a new SAMPLE (columns for new neurons)
%
%       'feedforward'
%           learningData.networkType
%           learningData.inputLayer_train
%           learningData.numNodes_hiddenLayers
%           learningData.neuronType_hiddenLayers
%           learningData.neuronType_outputLayer
%           learningData.outputLayer_train
%           learningData.learningRate
%           learningData.momentumTerm
%           learningData.tolerance
%           learningData.toleranceType
%           learningData.plot_info
%
%       'auto-encoder'
%           learningData.networkType
%           learningData.inputLayer_train
%           learningData.numNodes_hiddenLayers
%           learningData.neuronType_hiddenLayers
%           learningData.neuronType_outputLayer
%           learningData.learningRate
%           learningData.momentumTerm
%           learningData.tolerance
%           learningData.toleranceType
%           learningData.plot_info
%           
%

function [learningOutputs] = learnANN(learningData)
    
    switch learningData.networkType
        case 'feedforward'
                                
                                %---feedforward START--------
            inputLayer_train = learningData.inputLayer_train;
            numNodes_hiddenLayers = learningData.numNodes_hiddenLayers;
            neuronType_hiddenLayers = learningData.neuronType_hiddenLayers;
            neuronType_outputLayer = learningData.neuronType_outputLayer;
            outputLayer_train = learningData.outputLayer_train;
            learningRate = learningData.learningRate;
            momentumTerm = learningData.momentumTerm;
            tolerance = learningData.tolerance;
            toleranceType = learningData.toleranceType;
            plot_info = learningData.plot_info;

            %Initializing some variables----------------------
            numNodes_hiddenLayers = numNodes_hiddenLayers + 1; %for bias node
            inputLayer_train_bias = horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1));
            numLayers_hidden = max(size(numNodes_hiddenLayers));
            neuralActivation_Hidden_bias = cell(numLayers_hidden,1);

            for i = 1:numLayers_hidden
                neuralActivation_Hidden_bias{i} = ones(length(inputLayer_train(:,1)), numNodes_hiddenLayers(i)); %!!
            end

            numNodes_inputLayer = length(inputLayer_train(1,:))+1; %number of columns in inputLayer_train (training input layer) + bias node
            numNodes_outputLayer = length(outputLayer_train(1,:)); %number of columns in outputLayer_train (training output layer)
            weight_InputHidden = cell(numLayers_hidden,1);
            momentum_weight_InputHiddenUpdate = cell(numLayers_hidden,1);


            %Initialization of weights-----------------------
            weight_InputHidden{1} = 0.5*(1-2*rand(numNodes_inputLayer,numNodes_hiddenLayers(1)));
            for i = 2:numLayers_hidden
                weight_InputHidden{i} = 0.5*(1-2*rand(numNodes_hiddenLayers(i-1),numNodes_hiddenLayers(i))); %assign weights for hidden layers (including connection from inputs)
            end

            weight_HiddenOutput = 0.5*(1-2*rand(numNodes_hiddenLayers(end),numNodes_outputLayer)); %assign weights to neurons that connect from the hidden layer to the output layer
                %NB : weights are initialized randomly on the interval [-0.5 0.5]
            %-------------------------------------------------

            %Initialization of momentum weights---------------
            for i = 1:numLayers_hidden
                momentum_weight_InputHiddenUpdate{i} = zeros(size(weight_InputHidden{i}));
            end
            momentum_weight_HiddenOutputUpdate = zeros(size(weight_HiddenOutput));

            %Backpropagation algorithm------------------------
            errorPrev = 0; %for determining convergence
            convergence = tolerance+1;
            sse = [];
            integrate_InputHidden = cell(numLayers_hidden,1);
            neuralActivation_Hidden = cell(numLayers_hidden,1);
            delta_HiddenLayer = cell(numLayers_hidden,1);
            weightUpdate_InputHidden = cell(numLayers_hidden,1);
            
            while convergence > tolerance
                integrate_InputHidden{1} = integrate_input(horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1)),...
                    weight_InputHidden{1},numNodes_hiddenLayers(1)-1); %integrate information from input layer to hidden layer
                neuralActivation_Hidden{1} = neuralActivation(integrate_InputHidden{1},neuronType_hiddenLayers{1}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron

                for i = 2:numLayers_hidden
                    integrate_InputHidden{i} = integrate_input(horzcat(neuralActivation_Hidden{i-1},ones(length(neuralActivation_Hidden{i-1}(:,1)),1))...
                        ,weight_InputHidden{i},numNodes_hiddenLayers(i)-1); %integrate information from input layer to hidden layer
                    neuralActivation_Hidden{i} = neuralActivation(integrate_InputHidden{i},neuronType_hiddenLayers{i}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron
                end

                integrate_HiddenOutput = integrate_input(horzcat(neuralActivation_Hidden{numLayers_hidden},ones(length(neuralActivation_Hidden{numLayers_hidden}(:,1)),1))...
                    ,weight_HiddenOutput,numNodes_outputLayer); %integrate activations from hidden layer to the output layer
                neuralActivation_Output = neuralActivation(integrate_HiddenOutput,neuronType_outputLayer); %obtain the activations from the output neurons based on information integrated from the hidden layer and the type of the output neuron

                %Main backpropagation steps-------------------
                    %--computing errors
                delta_OutputLayer = (outputLayer_train - neuralActivation_Output).*...
                    neuralActivation_derivative(integrate_HiddenOutput,neuronType_outputLayer); %error in output layer

                delta_HiddenLayer{numLayers_hidden} = (delta_OutputLayer*weight_HiddenOutput') .* ...
                    neuralActivation_derivative(horzcat(integrate_InputHidden{numLayers_hidden},ones(length(integrate_InputHidden{numLayers_hidden}(:,1)),1))...
                    ,neuronType_hiddenLayers{numLayers_hidden}); %error in hidden layer (hence, backpropagation)

                for i = numLayers_hidden-1 : -1 : 1
                    delta_HiddenLayer{i} = (delta_HiddenLayer{i+1}*weight_InputHidden{i+1}') .* ...
                    neuralActivation_derivative(horzcat(integrate_InputHidden{i},ones(length(integrate_InputHidden{i}(:,1)),1))...
                    ,neuronType_hiddenLayers{i});
                end

                    %--compute weight update parameter
                for i = 1:numLayers_hidden
                    neuralActivation_Hidden_bias{i}(:,1:end-1) = neuralActivation_Hidden{i};
                end

                weightUpdate_InputHidden{1} = learningRate * inputLayer_train_bias' * delta_HiddenLayer{1}; %change in weights for neurons coming from the input layer to the hidden layer
                for i = 2:numLayers_hidden
                    weightUpdate_InputHidden{i} = learningRate * neuralActivation_Hidden_bias{i-1}' * delta_HiddenLayer{i};
                end
                weightUpdate_HiddenOutput = learningRate * neuralActivation_Hidden_bias{numLayers_hidden}' * delta_OutputLayer; %change in weights for neurons coming from the hidden layer to the output layer

                    %--update weights
                for i = 1:numLayers_hidden
                    weight_InputHidden{i} = weight_InputHidden{i} + weightUpdate_InputHidden{i};
                end
                weight_HiddenOutput = weight_HiddenOutput + weightUpdate_HiddenOutput;

                        %--add momentum
                        for i = 1:numLayers_hidden
                            weight_InputHidden{i} = weight_InputHidden{i} + momentumTerm * momentum_weight_InputHiddenUpdate{i};
                        end
                        weight_HiddenOutput = weight_HiddenOutput + momentumTerm * momentum_weight_HiddenOutputUpdate;

                        %reset momentum
                        momentum_weight_InputHiddenUpdate = weightUpdate_InputHidden;
                        %momentum_weight_HiddenOutputUpdate = weightUpdate_HiddenOutput;

                %Compute the error to determine convergence
                integrate_InputHidden{1} = integrate_input(horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1)),...
                    weight_InputHidden{1},numNodes_hiddenLayers(1)-1); %integrate information from input layer to hidden layer
                neuralActivation_Hidden{1} = neuralActivation(integrate_InputHidden{1},neuronType_hiddenLayers{1}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron

                for i = 2:numLayers_hidden
                    integrate_InputHidden{i} = integrate_input(horzcat(neuralActivation_Hidden{i-1},ones(length(neuralActivation_Hidden{i-1}(:,1)),1))...
                        ,weight_InputHidden{i},numNodes_hiddenLayers(i)-1); %integrate information from input layer to hidden layer
                    neuralActivation_Hidden{i} = neuralActivation(integrate_InputHidden{i},neuronType_hiddenLayers{i}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron
                end

                integrate_HiddenOutput = integrate_input(horzcat(neuralActivation_Hidden{numLayers_hidden},ones(length(neuralActivation_Hidden{numLayers_hidden}(:,1)),1))...
                    ,weight_HiddenOutput,numNodes_outputLayer); %integrate activations from hidden layer to the output layer
                neuralActivation_Output = neuralActivation(integrate_HiddenOutput,neuronType_outputLayer); %obtain the activations from the output neurons based on information integrated from the hidden layer and the type of the output neuron

                errorCurrent = sum(sum((outputLayer_train - neuralActivation_Output).^2)); %sum squared error
                errorCurrent
                switch toleranceType
                    case 'sse'
                        convergence = errorCurrent;
                    case 'absoluteDifference'
                        convergence = abs(errorCurrent - errorPrev); %convergence determined by absolute difference in the SSE
                        errorPrev = errorCurrent; %update the previous errors
                end
                sse = [sse errorCurrent];
                
            end

            if plot_info
                figure;
                plot(sse);
                figure;
                plot(neuralActivation_Output);
                hold on; plot(outputLayer_train,'.'); hold off;
            end
            
            learningOutputs = 1;
            
                         %%%%---feedforward END------------------------
            
            
                         
                         
                         
                         
                         
                         
                         
                         
        case 'auto-encoder'
            
                %--- auto-encoder START--------
            inputLayer_train = learningData.inputLayer_train;
            outputLayer_train = learningData.inputLayer_train;
            numNodes_hiddenLayers = learningData.numNodes_hiddenLayers;
            neuronType_hiddenLayers = learningData.neuronType_hiddenLayers;
            neuronType_outputLayer = learningData.neuronType_outputLayer;
            learningRate = learningData.learningRate;
            momentumTerm = learningData.momentumTerm;
            tolerance = learningData.tolerance;
            toleranceType = learningData.toleranceType;
            plot_info = learningData.plot_info;

            %Initializing some variables----------------------
            numNodes_hiddenLayers = numNodes_hiddenLayers + 1; %for bias node
            inputLayer_train_bias = horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1));
            numLayers_hidden = max(size(numNodes_hiddenLayers));
            neuralActivation_Hidden_bias = cell(numLayers_hidden,1);

            for i = 1:numLayers_hidden
                neuralActivation_Hidden_bias{i} = ones(length(inputLayer_train(:,1)), numNodes_hiddenLayers(i)); %!!
            end

            numNodes_inputLayer = length(inputLayer_train(1,:))+1; %number of columns in inputLayer_train (training input layer) + bias node
            numNodes_outputLayer = length(outputLayer_train(1,:)); %number of columns in outputLayer_train (training output layer)
            weight_InputHidden = cell(numLayers_hidden,1);
            momentum_weight_InputHiddenUpdate = cell(numLayers_hidden,1);


            %Initialization of weights-----------------------
            weight_InputHidden{1} = 0.5*(1-2*rand(numNodes_inputLayer,numNodes_hiddenLayers(1)));
            for i = 2:numLayers_hidden
                weight_InputHidden{i} = 0.5*(1-2*rand(numNodes_hiddenLayers(i-1),numNodes_hiddenLayers(i))); %assign weights for hidden layers (including connection from inputs)
            end

            weight_HiddenOutput = 0.5*(1-2*rand(numNodes_hiddenLayers(end),numNodes_outputLayer)); %assign weights to neurons that connect from the hidden layer to the output layer
                %NB : weights are initialized randomly on the interval [-0.5 0.5]
            %-------------------------------------------------

            %Initialization of momentum weights---------------
            for i = 1:numLayers_hidden
                momentum_weight_InputHiddenUpdate{i} = zeros(size(weight_InputHidden{i}));
            end
            momentum_weight_HiddenOutputUpdate = zeros(size(weight_HiddenOutput));

            %Backpropagation algorithm------------------------
            errorPrev = 0; %for determining convergence
            convergence = tolerance+1;
            sse = [];
            integrate_InputHidden = cell(numLayers_hidden,1);
            neuralActivation_Hidden = cell(numLayers_hidden,1);
            delta_HiddenLayer = cell(numLayers_hidden,1);
            weightUpdate_InputHidden = cell(numLayers_hidden,1);
            
            while convergence > tolerance
                integrate_InputHidden{1} = integrate_input(horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1)),...
                    weight_InputHidden{1},numNodes_hiddenLayers(1)-1); %integrate information from input layer to hidden layer
                neuralActivation_Hidden{1} = neuralActivation(integrate_InputHidden{1},neuronType_hiddenLayers{1}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron

                for i = 2:numLayers_hidden
                    integrate_InputHidden{i} = integrate_input(horzcat(neuralActivation_Hidden{i-1},ones(length(neuralActivation_Hidden{i-1}(:,1)),1))...
                        ,weight_InputHidden{i},numNodes_hiddenLayers(i)-1); %integrate information from input layer to hidden layer
                    neuralActivation_Hidden{i} = neuralActivation(integrate_InputHidden{i},neuronType_hiddenLayers{i}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron
                end

                integrate_HiddenOutput = integrate_input(horzcat(neuralActivation_Hidden{numLayers_hidden},ones(length(neuralActivation_Hidden{numLayers_hidden}(:,1)),1))...
                    ,weight_HiddenOutput,numNodes_outputLayer); %integrate activations from hidden layer to the output layer
                neuralActivation_Output = neuralActivation(integrate_HiddenOutput,neuronType_outputLayer); %obtain the activations from the output neurons based on information integrated from the hidden layer and the type of the output neuron

                %Main backpropagation steps-------------------
                    %--computing errors
                delta_OutputLayer = (outputLayer_train - neuralActivation_Output).*...
                    neuralActivation_derivative(integrate_HiddenOutput,neuronType_outputLayer); %error in output layer

                delta_HiddenLayer{numLayers_hidden} = (delta_OutputLayer*weight_HiddenOutput') .* ...
                    neuralActivation_derivative(horzcat(integrate_InputHidden{numLayers_hidden},ones(length(integrate_InputHidden{numLayers_hidden}(:,1)),1))...
                    ,neuronType_hiddenLayers{numLayers_hidden}); %error in hidden layer (hence, backpropagation)

                for i = numLayers_hidden-1 : -1 : 1
                    delta_HiddenLayer{i} = (delta_HiddenLayer{i+1}*weight_InputHidden{i+1}') .* ...
                    neuralActivation_derivative(horzcat(integrate_InputHidden{i},ones(length(integrate_InputHidden{i}(:,1)),1))...
                    ,neuronType_hiddenLayers{i});
                end

                    %--compute weight update parameter
                for i = 1:numLayers_hidden
                    neuralActivation_Hidden_bias{i}(:,1:end-1) = neuralActivation_Hidden{i};
                end

                weightUpdate_InputHidden{1} = learningRate * inputLayer_train_bias' * delta_HiddenLayer{1}; %change in weights for neurons coming from the input layer to the hidden layer
                for i = 2:numLayers_hidden
                    weightUpdate_InputHidden{i} = learningRate * neuralActivation_Hidden_bias{i-1}' * delta_HiddenLayer{i};
                end
                weightUpdate_HiddenOutput = learningRate * neuralActivation_Hidden_bias{numLayers_hidden}' * delta_OutputLayer; %change in weights for neurons coming from the hidden layer to the output layer

                    %--update weights
                for i = 1:numLayers_hidden
                    weight_InputHidden{i} = weight_InputHidden{i} + weightUpdate_InputHidden{i};
                end
                weight_HiddenOutput = weight_HiddenOutput + weightUpdate_HiddenOutput;

                        %--add momentum
                        for i = 1:numLayers_hidden
                            weight_InputHidden{i} = weight_InputHidden{i} + momentumTerm * momentum_weight_InputHiddenUpdate{i};
                        end
                        weight_HiddenOutput = weight_HiddenOutput + momentumTerm * momentum_weight_HiddenOutputUpdate;

                        %reset momentum
                        momentum_weight_InputHiddenUpdate = weightUpdate_InputHidden;
                        %momentum_weight_HiddenOutputUpdate = weightUpdate_HiddenOutput;

                %Compute the error to determine convergence
                integrate_InputHidden{1} = integrate_input(horzcat(inputLayer_train,ones(length(inputLayer_train(:,1)),1)),...
                    weight_InputHidden{1},numNodes_hiddenLayers(1)-1); %integrate information from input layer to hidden layer
                neuralActivation_Hidden{1} = neuralActivation(integrate_InputHidden{1},neuronType_hiddenLayers{1}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron

                for i = 2:numLayers_hidden
                    integrate_InputHidden{i} = integrate_input(horzcat(neuralActivation_Hidden{i-1},ones(length(neuralActivation_Hidden{i-1}(:,1)),1))...
                        ,weight_InputHidden{i},numNodes_hiddenLayers(i)-1); %integrate information from input layer to hidden layer
                    neuralActivation_Hidden{i} = neuralActivation(integrate_InputHidden{i},neuronType_hiddenLayers{i}); %obtain activations from neurons in hidden layer based on the integrated input and the type of neuron
                end

                integrate_HiddenOutput = integrate_input(horzcat(neuralActivation_Hidden{numLayers_hidden},ones(length(neuralActivation_Hidden{numLayers_hidden}(:,1)),1))...
                    ,weight_HiddenOutput,numNodes_outputLayer); %integrate activations from hidden layer to the output layer
                neuralActivation_Output = neuralActivation(integrate_HiddenOutput,neuronType_outputLayer); %obtain the activations from the output neurons based on information integrated from the hidden layer and the type of the output neuron

                errorCurrent = sum(sum((outputLayer_train - neuralActivation_Output).^2)); %sum squared error

                switch toleranceType
                    case 'sse'
                        convergence = errorCurrent;
                    case 'absoluteDifference'
                        convergence = abs(errorCurrent - errorPrev); %convergence determined by absolute difference in the SSE
                        errorPrev = errorCurrent; %update the previous errors
                end
                sse = [sse errorCurrent];

            end

            if plot_info
                figure;
                plot(sse);
                figure;
                plot(neuralActivation_Output);
                hold on; plot(outputLayer_train,'.'); hold off;
            end
                %--- auto-encoder END----------------------------
    end
    
    
    
    
    %------Helper functions------------------------------------------------
    function z = integrate_input(input,weight,n)
        z = input*weight(:,1:n);
    
    function a = neuralActivation(z,neuronType)
        switch neuronType
            case 'linear'
                a = z;
                
            case 'sigmoid'
                a = 1./(1 + exp(-z));
            
            case 'tanh'
                a = 2./(1 + exp(-2.*z))-1;
            
            case 'binary'
                a = (z >= 0);
                
            case 'arctan'
                a = atan(z);
                
            case 'rectifiedLinear'
                a = (z >= 0).*z;
                
            case 'softPlus'
                a = log(1 + exp(z));
                
            case 'bentLinear'
                a = ( (sqrt(z.^2 + 1) - 1) / 2 ) + z;
                
            case 'sinusoid'
                a = sin(z);
                
            case 'sinc'
                a = (z ~= 0).*sin(z)./z;
                a(a==0) = 1;
                
            case 'gaussian'
                a = exp(-z.^2);
                
            case 'softmax'
                numNeurons_Layer = length(z(1,:));
                sumZ = repmat(sum(exp(z),2),[1, numNeurons_Layer]);
                a = exp(z)./(sumZ);
        end

    function delta = neuralActivation_derivative(z,neuronType)
            
        switch neuronType
            case 'linear'
                delta = ones(size(z));
                
            case 'sigmoid'
                x = neuralActivation(z,'sigmoid');
                delta = x.*(1-x);
                
            case 'tanh'
                x = neuralActivation(z,'tanh');
                delta = 1 - x.^2;
                
            case 'arctan'
                z = z + bias;
                delta = 1./(z.^2 + 1);
                
            case 'binary'
                %delta = (z >= 0); %delta = double(delta);
                %delta(delta==1) = NaN; %---for application purposes avoid
                
                delta = ones(size(z)); %faux
                
            case 'rectifiedLinear'
                delta = (z >= 0);
                
            case 'softPlus'
                delta = neuralActivation(z,'sigmoid');
                
            case 'bentLinear'
                delta = z./(2.*sqrt(z.^2 + 1)) + 1;
                
            case 'sinusoid'
                delta = cos(z);
                
            case 'sinc'
                delta = (z ~= 0).*( (cos(z)./z) - (sin(z)./(z.^2)) );
                
            case 'gaussian'
                delta = -2.*z.*exp(-z.^2);
                
            case 'softmax'
                x = neuralActivation(z,'softmax');
                delta = x.*(1-x);
        end