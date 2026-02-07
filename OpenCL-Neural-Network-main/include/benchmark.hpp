#pragma once

#include <vector>
#include <string>
#include <chrono>
#include "matrix_ops.hpp"
#include "activation_functions.hpp"

class Benchmark {
public:
    Benchmark(const OpenCLContext& context);
    ~Benchmark() = default;
    
    // Matrix multiplication benchmarks
    void benchmarkMatrixMultiplication(int minSize, int maxSize, int step);
    void benchmarkMatrixMultiplicationDetailed(int size);
    
    // Activation function benchmarks
    void benchmarkActivationFunctions(int vectorSize);
    void benchmarkActivationFunctionsDetailed(int vectorSize);
    
    // Combined neural network layer benchmark
    void benchmarkNeuralLayer(int inputSize, int hiddenSize, int outputSize);
    
    // Utility functions
    void printResults() const;
    void saveResultsToFile(const std::string& filename) const;
    
private:
    MatrixOps matrixOps_;
    ActivationFunctions activationFuncs_;
    
    struct BenchmarkResult {
        std::string operation;
        std::string implementation;
        int size;
        double time_ms;
        double speedup;
    };
    
    std::vector<BenchmarkResult> results_;
    
    template<typename Func>
    double measureTime(Func func, int iterations = 1);
    
    void addResult(const std::string& operation, const std::string& implementation, 
                  int size, double time_ms, double speedup = 1.0);
};
