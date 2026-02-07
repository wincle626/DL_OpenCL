#pragma once

#include <vector>
#include <memory>
#include "opencl_utils.hpp"

class MatrixOps {
public:
    MatrixOps(const OpenCLContext& context);
    ~MatrixOps() = default;
    
    // Matrix multiplication: C = A * B
    std::vector<float> multiplyGPU(const std::vector<float>& A, const std::vector<float>& B, 
                                  int rowsA, int colsA, int colsB);
    
    std::vector<float> multiplyCPU(const std::vector<float>& A, const std::vector<float>& B, 
                                  int rowsA, int colsA, int colsB);
    
    // Matrix addition: C = A + B
    std::vector<float> addGPU(const std::vector<float>& A, const std::vector<float>& B, int size);
    
    // Matrix transpose
    std::vector<float> transposeGPU(const std::vector<float>& A, int rows, int cols);
    
    // Utility functions
    std::vector<float> createRandomMatrix(int rows, int cols, float min = -1.0f, float max = 1.0f);
    void printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name = "");
    bool compareMatrices(const std::vector<float>& A, const std::vector<float>& B, float tolerance = 1e-5f);
    
private:
    const OpenCLContext& context_;
    std::unique_ptr<OpenCLKernel> matrixMultKernel_;
    std::unique_ptr<OpenCLKernel> matrixAddKernel_;
    std::unique_ptr<OpenCLKernel> matrixTransposeKernel_;
    
    void initializeKernels();
    void setMatrixMultArgs(cl_kernel kernel, cl_mem A, cl_mem B, cl_mem C, 
                          int rowsA, int colsA, int colsB);
};
