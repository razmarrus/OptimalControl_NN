function [x] = ReluLayer(x)
    if x < 0
       x = 0;
    end
end

function [x] = TanhLayer(x)
    x = tanh(x);
end

function [x] =  SoftPlusLayer(x)
    %softplusLayer('Name','stddev')
    x = log(exp(x) + 1);
end


function [x] = MaxPoolingLayer(x) %, pool_size, strides, padding)
    %maxPooling2dLayer
    x = max(x); %there also should be reshape
end

%{function [x] = DropOutLayer(x) %, p, dropout_rate) 
%    if p < droput_rate
        %skip connection
%    end
%end


function [x_next] = FullyConnectedStepForward(x, vars) %BinaryFullyConnectedLayer(x)
    %softplusLayer('Name','stddev')
    dim = size(A); %returns a row vector whose elements are the lengths of the corresponding dimensions of x
    dim_row_size = size(dim);
    x_in_dim = 1;
    for dim_number = 2:1:dim_row_size %iterating through the array from second element
        current_dim_product = prod(x,dim_number);
        x_in_dim =  x_in_dim + current_dim_product;
    end
   %first_row_var_array_dim = size(A, 1);
   %w_in_dim = first_row_var_array_dim(1);
   x_next = x * vars(1);
   
end

function [H, dHdw] = H_and_grad(x, p,vars) %, pool_size, strides, padding)

    dHdp = StepForward(x, vars);
    H = sum(p * dHdp);
    dHdw = gradient(H, self.vars);
end
        

function [x_next] = ConvolutionStepForward(x, vars) %BinaryFullyConnectedLayer(x)
    dim = size(vars); %returns a row vector whose elements are the lengths of the corresponding dimensions of x
    dim_row_size = size(dim);
    
    x_in_dim = x(length(x));
    w_in_dim = self.vars[0].get_shape().as_list()[2]
    
    
    for dim_number = 2:1:dim_row_size %iterating through the array from second element
        current_dim_product = prod(x,dim_number);
        x_in_dim =  x_in_dim + current_dim_product;
    end
   %first_row_var_array_dim = size(A, 1);
   %w_in_dim = first_row_var_array_dim(1);
   x_next = x * vars(1);
   
end

function [x_next] =  BatchNormLayerStepForward(x, vars, axes, epsilon) 
    x_in_dim = size(x); %returns a row vector whose elements are the lengths of the corresponding dimensions of x
    x_in_dim(axes) = 1;
    
    scale_dim = size(vars(0));
    offset_dim = size(vars(1));
    try
       and((x_in_dim == scale_dim),(in_dim == offset_dim))
    catch 
       warning('Input/vars shape inconsistent.');
    end
    M = mean(A);
    V = var(A);
    %TODO: updating mean and variance for this layer, if they are None%
    x_normalized = (x - M) / sqrt(epsilon + V);
    x_next = vars(0)*x_normalized + vars(1);
    
end





