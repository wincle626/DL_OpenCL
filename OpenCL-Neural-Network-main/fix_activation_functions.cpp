#include "activation_functions.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

ActivationFunctions::ActivationFunctions(const OpenCLContext& context) : context_(context) {
    initializeKernels();
}

void ActivationFunctions::initializeKernels() {
    try {
        std::string activationKernelSource = readKernelSource("kernels/activation_functions.cl");
        
        reluKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "relu");
        sigmoidKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "sigmoid");
        tanhKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "tanh");
        reluDerivativeKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "relu_derivative");
        sigmoidDerivativeKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "sigmoid_derivative");
        tanhDerivativeKernel_ = std::make_unique<OpenCLKernel>(context_, activationKernelSource, "tanh_derivative");
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing activation function kernels: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> ActivationFunctions::reluGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = reluKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing ReLU kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

std::vector<float> ActivationFunctions::reluCPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::max(0.0f, input[i]);
    }
    return output;
}

std::vector<float> ActivationFunctions::sigmoidGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = sigmoidKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing Sigmoid kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

std::vector<float> ActivationFunctions::sigmoidCPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return output;
}

std::vector<float> ActivationFunctions::tanhGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = tanhKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing Tanh kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

std::vector<float> ActivationFunctions::tanhCPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = std::tanh(input[i]);
    }
    return output;
}

std::vector<float> ActivationFunctions::reluDerivativeGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = reluDerivativeKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing ReLU derivative kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

std::vector<float> ActivationFunctions::sigmoidDerivativeGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = sigmoidDerivativeKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing Sigmoid derivative kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

std::vector<float> ActivationFunctions::tanhDerivativeGPU(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    OpenCLBuffer bufferInput(context_, input.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferOutput(context_, output.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferInput.writeData(input.data());
    
    cl_kernel kernel = tanhDerivativeKernel_->getKernel();
    cl_mem inputBuffer = bufferInput.getBuffer();
    cl_mem outputBuffer = bufferOutput.getBuffer();
    int size = static_cast<int>(input.size());
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 2");
    
    size_t globalWorkSize = input.size();
    size_t localWorkSize = 256;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing Tanh derivative kernel");
    
    bufferOutput.readData(output.data());
    return output;
}

void ActivationFunctions::printVector(const std::vector<float>& vec, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << " (" << vec.size() << " elements):" << std::endl;
    }
    
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << vec[i];
        if ((i + 1) % 8 == 0) std::cout << std::endl;
    }
    if (vec.size() % 8 != 0) std::cout << std::endl;
    std::cout << std::endl;
}

bool ActivationFunctions::compareVectors(const std::vector<float>& A, const std::vector<float>& B, float tolerance) {
    if (A.size() != B.size()) {
        return false;
    }
    
    for (size_t i = 0; i < A.size(); i++) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}
