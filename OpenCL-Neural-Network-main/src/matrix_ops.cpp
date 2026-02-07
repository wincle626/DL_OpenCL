#include "matrix_ops.hpp"
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>

MatrixOps::MatrixOps(const OpenCLContext& context) : context_(context) {
    initializeKernels();
}

void MatrixOps::initializeKernels() {
    try {
        std::string matrixKernelSource = readKernelSource("kernels/matrix_ops.cl");
        
        matrixMultKernel_ = std::make_unique<OpenCLKernel>(context_, matrixKernelSource, "matrix_multiply");
        matrixAddKernel_ = std::make_unique<OpenCLKernel>(context_, matrixKernelSource, "matrix_add");
        matrixTransposeKernel_ = std::make_unique<OpenCLKernel>(context_, matrixKernelSource, "matrix_transpose");
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing kernels: " << e.what() << std::endl;
        throw;
    }
}

std::vector<float> MatrixOps::multiplyGPU(const std::vector<float>& A, const std::vector<float>& B, 
                                         int rowsA, int colsA, int colsB) {
    int rowsC = rowsA;
    int colsC = colsB;
    std::vector<float> C(rowsC * colsC);
    
    // Create OpenCL buffers
    OpenCLBuffer bufferA(context_, A.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferB(context_, B.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferC(context_, C.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    // Copy data to device
    bufferA.writeData(A.data());
    bufferB.writeData(B.data());
    
    // Set kernel arguments
    cl_kernel kernel = matrixMultKernel_->getKernel();
    setMatrixMultArgs(kernel, bufferA.getBuffer(), bufferB.getBuffer(), bufferC.getBuffer(), 
                      rowsA, colsA, colsB);
    
    // Calculate work group sizes
    size_t globalWorkSize[2] = {static_cast<size_t>(colsB), static_cast<size_t>(rowsA)};
    // Use NULL for local work size to let OpenCL choose optimal size
    size_t* localWorkSize = nullptr;
    
    // Execute kernel
    cl_int error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 2, nullptr, 
                                          globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing matrix multiplication kernel");
    
    // Read result back
    bufferC.readData(C.data());
    
    return C;
}

std::vector<float> MatrixOps::multiplyCPU(const std::vector<float>& A, const std::vector<float>& B, 
                                         int rowsA, int colsA, int colsB) {
    int rowsC = rowsA;
    int colsC = colsB;
    std::vector<float> C(rowsC * colsC, 0.0f);
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsC + j] = sum;
        }
    }
    
    return C;
}

std::vector<float> MatrixOps::addGPU(const std::vector<float>& A, const std::vector<float>& B, int size) {
    std::vector<float> C(size);
    
    OpenCLBuffer bufferA(context_, A.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferB(context_, B.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferC(context_, C.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferA.writeData(A.data());
    bufferB.writeData(B.data());
    
    cl_kernel kernel = matrixAddKernel_->getKernel();
    cl_mem bufferA_mem = bufferA.getBuffer();
    cl_mem bufferB_mem = bufferB.getBuffer();
    cl_mem bufferC_mem = bufferC.getBuffer();
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA_mem);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB_mem);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC_mem);
    checkOpenCLError(error, "Setting kernel argument 2");
    error = clSetKernelArg(kernel, 3, sizeof(int), &size);
    checkOpenCLError(error, "Setting kernel argument 3");
    
    size_t globalWorkSize = size;
    // Use NULL for local work size to let OpenCL choose optimal size
    size_t* localWorkSize = nullptr;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 1, nullptr, 
                                   &globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing matrix addition kernel");
    
    bufferC.readData(C.data());
    return C;
}

std::vector<float> MatrixOps::transposeGPU(const std::vector<float>& A, int rows, int cols) {
    std::vector<float> B(rows * cols);
    
    OpenCLBuffer bufferA(context_, A.size() * sizeof(float), CL_MEM_READ_ONLY);
    OpenCLBuffer bufferB(context_, B.size() * sizeof(float), CL_MEM_WRITE_ONLY);
    
    bufferA.writeData(A.data());
    
    cl_kernel kernel = matrixTransposeKernel_->getKernel();
    cl_mem bufferA_mem = bufferA.getBuffer();
    cl_mem bufferB_mem = bufferB.getBuffer();
    
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA_mem);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB_mem);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(int), &rows);
    checkOpenCLError(error, "Setting kernel argument 2");
    error = clSetKernelArg(kernel, 3, sizeof(int), &cols);
    checkOpenCLError(error, "Setting kernel argument 3");
    
    size_t globalWorkSize[2] = {static_cast<size_t>(cols), static_cast<size_t>(rows)};
    // Use NULL for local work size to let OpenCL choose optimal size
    size_t* localWorkSize = nullptr;
    
    error = clEnqueueNDRangeKernel(context_.getCommandQueue(), kernel, 2, nullptr, 
                                   globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    checkOpenCLError(error, "Enqueuing matrix transpose kernel");
    
    bufferB.readData(B.data());
    return B;
}

std::vector<float> MatrixOps::createRandomMatrix(int rows, int cols, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    std::vector<float> matrix(rows * cols);
    for (auto& val : matrix) {
        val = dis(gen);
    }
    return matrix;
}

void MatrixOps::printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    if (!name.empty()) {
        std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << matrix[i * cols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

bool MatrixOps::compareMatrices(const std::vector<float>& A, const std::vector<float>& B, float tolerance) {
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

void MatrixOps::setMatrixMultArgs(cl_kernel kernel, cl_mem A, cl_mem B, cl_mem C, 
                                  int rowsA, int colsA, int colsB) {
    cl_int error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    checkOpenCLError(error, "Setting kernel argument 0");
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    checkOpenCLError(error, "Setting kernel argument 1");
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    checkOpenCLError(error, "Setting kernel argument 2");
    error = clSetKernelArg(kernel, 3, sizeof(int), &rowsA);
    checkOpenCLError(error, "Setting kernel argument 3");
    error = clSetKernelArg(kernel, 4, sizeof(int), &colsA);
    checkOpenCLError(error, "Setting kernel argument 4");
    error = clSetKernelArg(kernel, 5, sizeof(int), &colsB);
    checkOpenCLError(error, "Setting kernel argument 5");
}
