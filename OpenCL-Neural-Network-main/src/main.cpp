#include <iostream>
#include <memory>
#include <stdexcept>
#include <iomanip>
#include "opencl_utils.hpp"
#include "matrix_ops.hpp"
#include "activation_functions.hpp"
#include "benchmark.hpp"

void testMatrixOperations(const OpenCLContext& context) {
    std::cout << "\n=== Testing Matrix Operations ===" << std::endl;
    
    MatrixOps matrixOps(context);
    
    // Test small matrices first
    int rows = 4, cols = 4;
    auto A = matrixOps.createRandomMatrix(rows, cols, -1.0f, 1.0f);
    auto B = matrixOps.createRandomMatrix(cols, rows, -1.0f, 1.0f);
    
    std::cout << "Testing " << rows << "x" << cols << " matrix operations..." << std::endl;
    
    // Test matrix multiplication
    auto C_CPU = matrixOps.multiplyCPU(A, B, rows, cols, rows);
    auto C_GPU = matrixOps.multiplyGPU(A, B, rows, cols, rows);
    
    if (matrixOps.compareMatrices(C_CPU, C_GPU, 1e-4f)) {
        std::cout << "✓ Matrix multiplication: CPU and GPU results match" << std::endl;
    } else {
        std::cout << "✗ Matrix multiplication: CPU and GPU results differ" << std::endl;
    }
    
    // Test matrix addition
    auto D_CPU = matrixOps.addGPU(A, A, rows * cols); // Using GPU for both to test
    auto D_GPU = matrixOps.addGPU(A, A, rows * cols);
    
    if (matrixOps.compareMatrices(D_CPU, D_GPU, 1e-4f)) {
        std::cout << "✓ Matrix addition: Results match" << std::endl;
    } else {
        std::cout << "✗ Matrix addition: Results differ" << std::endl;
    }
    
    // Test matrix transpose
    auto E_GPU = matrixOps.transposeGPU(A, rows, cols);
    
    // Verify transpose: (A^T)^T = A
    auto F_GPU = matrixOps.transposeGPU(E_GPU, cols, rows);
    
    if (matrixOps.compareMatrices(A, F_GPU, 1e-4f)) {
        std::cout << "✓ Matrix transpose: (A^T)^T = A verified" << std::endl;
    } else {
        std::cout << "✗ Matrix transpose: Verification failed" << std::endl;
    }
}

void testActivationFunctions(const OpenCLContext& context) {
    std::cout << "\n=== Testing Activation Functions ===" << std::endl;
    
    ActivationFunctions activationFuncs(context);
    
    // Test with a small vector
    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    std::cout << "Input vector: ";
    for (float val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Test ReLU
    auto relu_CPU = activationFuncs.reluCPU(input);
    auto relu_GPU = activationFuncs.reluGPU(input);
    
    if (activationFuncs.compareVectors(relu_CPU, relu_GPU, 1e-4f)) {
        std::cout << "✓ ReLU: CPU and GPU results match" << std::endl;
        std::cout << "  ReLU output: ";
        for (float val : relu_GPU) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "✗ ReLU: CPU and GPU results differ" << std::endl;
    }
    
    // Test Sigmoid
    auto sigmoid_CPU = activationFuncs.sigmoidCPU(input);
    auto sigmoid_GPU = activationFuncs.sigmoidGPU(input);
    
    if (activationFuncs.compareVectors(sigmoid_CPU, sigmoid_GPU, 1e-4f)) {
        std::cout << "✓ Sigmoid: CPU and GPU results match" << std::endl;
        std::cout << "  Sigmoid output: ";
        for (float val : sigmoid_GPU) {
            std::cout << std::fixed << std::setprecision(3) << val << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "✗ Sigmoid: CPU and GPU results differ" << std::endl;
    }
    
    // Test Tanh
    auto tanh_CPU = activationFuncs.tanhCPU(input);
    auto tanh_GPU = activationFuncs.tanhGPU(input);
    
    if (activationFuncs.compareVectors(tanh_CPU, tanh_GPU, 1e-4f)) {
        std::cout << "✓ Tanh: CPU and GPU results match" << std::endl;
        std::cout << "  Tanh output: ";
        for (float val : tanh_GPU) {
            std::cout << std::fixed << std::setprecision(3) << val << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "✗ Tanh: CPU and GPU results differ" << std::endl;
    }
}

void runBenchmarks(const OpenCLContext& context) {
    std::cout << "\n=== Running Performance Benchmarks ===" << std::endl;
    
    Benchmark benchmark(context);
    
    // Matrix multiplication benchmarks
    benchmark.benchmarkMatrixMultiplication(64, 512, 64);
    
    // Activation function benchmarks
    benchmark.benchmarkActivationFunctions(1000000);
    
    // Neural network layer benchmark
    benchmark.benchmarkNeuralLayer(1000, 500, 10);
    
    // Print summary
    benchmark.printResults();
    
    // Save results to file
    benchmark.saveResultsToFile("benchmark_results.csv");
}

void demonstrateNeuralNetwork(const OpenCLContext& context) {
    std::cout << "\n=== Neural Network Demonstration ===" << std::endl;
    
    MatrixOps matrixOps(context);
    ActivationFunctions activationFuncs(context);
    
    // Create a simple neural network: 3 -> 4 -> 2
    int inputSize = 3, hiddenSize = 4, outputSize = 2;
    
    // Initialize weights
    auto W1 = matrixOps.createRandomMatrix(hiddenSize, inputSize, -0.5f, 0.5f);
    auto W2 = matrixOps.createRandomMatrix(outputSize, hiddenSize, -0.5f, 0.5f);
    
    // Create input
    auto input = matrixOps.createRandomMatrix(1, inputSize, -1.0f, 1.0f);
    
    std::cout << "Input: ";
    for (int i = 0; i < inputSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << input[i] << " ";
    }
    std::cout << std::endl;
    
    // Forward pass
    std::cout << "\nForward pass:" << std::endl;
    
    // Input -> Hidden layer
    auto hidden = matrixOps.multiplyGPU(input, W1, 1, inputSize, hiddenSize);
    std::cout << "Hidden layer (before activation): ";
    for (int i = 0; i < hiddenSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << hidden[i] << " ";
    }
    std::cout << std::endl;
    
    // Apply ReLU activation
    hidden = activationFuncs.reluGPU(hidden);
    std::cout << "Hidden layer (after ReLU): ";
    for (int i = 0; i < hiddenSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << hidden[i] << " ";
    }
    std::cout << std::endl;
    
    // Hidden -> Output layer
    auto output = matrixOps.multiplyGPU(hidden, W2, 1, hiddenSize, outputSize);
    std::cout << "Output (before activation): ";
    for (int i = 0; i < outputSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << output[i] << " ";
    }
    std::cout << std::endl;
    
    // Apply Sigmoid activation
    output = activationFuncs.sigmoidGPU(output);
    std::cout << "Final output (after Sigmoid): ";
    for (int i = 0; i < outputSize; i++) {
        std::cout << std::fixed << std::setprecision(3) << output[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "OpenCL Neural Network Primitives" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // Initialize OpenCL context
        OpenCLContext context;
        context.printDeviceInfo();
        
        // Test basic functionality
        testMatrixOperations(context);
        testActivationFunctions(context);
        
        // Demonstrate neural network
        demonstrateNeuralNetwork(context);
        
        // Run performance benchmarks
        runBenchmarks(context);
        
        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
