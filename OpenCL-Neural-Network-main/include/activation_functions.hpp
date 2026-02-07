#pragma once

#include <vector>
#include <memory>
#include "opencl_utils.hpp"

class ActivationFunctions {
public:
    ActivationFunctions(const OpenCLContext& context);
    ~ActivationFunctions() = default;
    
    // ReLU: f(x) = max(0, x)
    std::vector<float> reluGPU(const std::vector<float>& input);
    std::vector<float> reluCPU(const std::vector<float>& input);
    
    // Sigmoid: f(x) = 1 / (1 + e^(-x))
    std::vector<float> sigmoidGPU(const std::vector<float>& input);
    std::vector<float> sigmoidCPU(const std::vector<float>& input);
    
    // Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    std::vector<float> tanhGPU(const std::vector<float>& input);
    std::vector<float> tanhCPU(const std::vector<float>& input);
    
    // Derivative functions for backpropagation
    std::vector<float> reluDerivativeGPU(const std::vector<float>& input);
    std::vector<float> sigmoidDerivativeGPU(const std::vector<float>& input);
    std::vector<float> tanhDerivativeGPU(const std::vector<float>& input);
    
    // Utility functions
    void printVector(const std::vector<float>& vec, const std::string& name = "");
    bool compareVectors(const std::vector<float>& A, const std::vector<float>& B, float tolerance = 1e-5f);
    
private:
    const OpenCLContext& context_;
    std::unique_ptr<OpenCLKernel> reluKernel_;
    std::unique_ptr<OpenCLKernel> sigmoidKernel_;
    std::unique_ptr<OpenCLKernel> tanhKernel_;
    std::unique_ptr<OpenCLKernel> reluDerivativeKernel_;
    std::unique_ptr<OpenCLKernel> sigmoidDerivativeKernel_;
    std::unique_ptr<OpenCLKernel> tanhDerivativeKernel_;
    
    void initializeKernels();
};
