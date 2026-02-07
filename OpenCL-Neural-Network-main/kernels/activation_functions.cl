// ReLU activation function: f(x) = max(0, x)
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = fmax(0.0f, input[idx]);
    }
}

// ReLU derivative: f'(x) = 1 if x > 0, else 0
__kernel void relu_derivative(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float exp_val = exp(-input[idx]);
        output[idx] = 1.0f / (1.0f + exp_val);
    }
}

// Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
__kernel void sigmoid_derivative(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float exp_val = exp(-input[idx]);
        float sigmoid_val = 1.0f / (1.0f + exp_val);
        output[idx] = sigmoid_val * (1.0f - sigmoid_val);
    }
}

// Tanh activation function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
__kernel void tanh(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float exp_pos = exp(input[idx]);
        float exp_neg = exp(-input[idx]);
        output[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}

// Tanh derivative: f'(x) = 1 - f(x)^2
__kernel void tanh_derivative(
    __global const float* input,
    __global float* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float exp_pos = exp(input[idx]);
        float exp_neg = exp(-input[idx]);
        float tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg);
        output[idx] = 1.0f - tanh_val * tanh_val;
    }
}

// Vectorized activation functions for better performance
__kernel void relu_vectorized(
    __global const float4* input,
    __global float4* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float4 in = input[idx];
        output[idx] = (float4)(fmax(0.0f, in.x), fmax(0.0f, in.y), 
                               fmax(0.0f, in.z), fmax(0.0f, in.w));
    }
}

__kernel void sigmoid_vectorized(
    __global const float4* input,
    __global float4* output,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        float4 in = input[idx];
        float4 exp_neg = exp(-in);
        output[idx] = (float4)(1.0f / (1.0f + exp_neg.x),
                               1.0f / (1.0f + exp_neg.y),
                               1.0f / (1.0f + exp_neg.z),
                               1.0f / (1.0f + exp_neg.w));
    }
}

// Softmax activation function for classification
__kernel void softmax(
    __global const float* input,
    __global float* output,
    __local float* local_max,
    __local float* local_sum,
    const int size
) {
    int idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int group_size = get_local_size(0);
    
    // Find local maximum
    local_max[local_idx] = input[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (local_idx < offset) {
            local_max[local_idx] = fmax(local_max[local_idx], local_max[local_idx + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute exp(x - max) and local sum
    float exp_val = exp(input[idx] - local_max[0]);
    output[idx] = exp_val;
    local_sum[local_idx] = exp_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce sum
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        if (local_idx < offset) {
            local_sum[local_idx] += local_sum[local_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Normalize
    if (local_idx == 0) {
        output[idx] = exp_val / local_sum[0];
    }
}
