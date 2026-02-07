#include "benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

Benchmark::Benchmark(const OpenCLContext& context) 
    : matrixOps_(context), activationFuncs_(context) {
}

void Benchmark::benchmarkMatrixMultiplication(int minSize, int maxSize, int step) {
    std::cout << "\n=== Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Testing sizes from " << minSize << "x" << minSize << " to " << maxSize << "x" << maxSize << std::endl;
    
    for (int size = minSize; size <= maxSize; size += step) {
        benchmarkMatrixMultiplicationDetailed(size);
    }
}

void Benchmark::benchmarkMatrixMultiplicationDetailed(int size) {
    std::cout << "\nMatrix size: " << size << "x" << size << std::endl;
    
    // Create test matrices
    auto A = matrixOps_.createRandomMatrix(size, size, -1.0f, 1.0f);
    auto B = matrixOps_.createRandomMatrix(size, size, -1.0f, 1.0f);
    
    // Benchmark CPU
    double cpuTime = measureTime([&]() {
        matrixOps_.multiplyCPU(A, B, size, size, size);
    }, 3);
    
    // Benchmark GPU
    double gpuTime = measureTime([&]() {
        matrixOps_.multiplyGPU(A, B, size, size, size);
    }, 3);
    
    double speedup = cpuTime / gpuTime;
    
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpuTime << " ms" << std::endl;
    std::cout << "  GPU: " << std::fixed << std::setprecision(2) << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    addResult("Matrix Multiplication", "CPU", size, cpuTime);
    addResult("Matrix Multiplication", "GPU", size, gpuTime, speedup);
}

void Benchmark::benchmarkActivationFunctions(int vectorSize) {
    std::cout << "\n=== Activation Functions Benchmark ===" << std::endl;
    std::cout << "Vector size: " << vectorSize << std::endl;
    
    benchmarkActivationFunctionsDetailed(vectorSize);
}

void Benchmark::benchmarkActivationFunctionsDetailed(int vectorSize) {
    // Create test vector
    auto input = matrixOps_.createRandomMatrix(1, vectorSize, -5.0f, 5.0f);
    
    // Benchmark ReLU
    std::cout << "\nReLU:" << std::endl;
    double cpuTime = measureTime([&]() {
        activationFuncs_.reluCPU(input);
    }, 10);
    double gpuTime = measureTime([&]() {
        activationFuncs_.reluGPU(input);
    }, 10);
    double speedup = cpuTime / gpuTime;
    
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpuTime << " ms" << std::endl;
    std::cout << "  GPU: " << std::fixed << std::setprecision(2) << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    addResult("ReLU", "CPU", vectorSize, cpuTime);
    addResult("ReLU", "GPU", vectorSize, gpuTime, speedup);
    
    // Benchmark Sigmoid
    std::cout << "\nSigmoid:" << std::endl;
    cpuTime = measureTime([&]() {
        activationFuncs_.sigmoidCPU(input);
    }, 10);
    gpuTime = measureTime([&]() {
        activationFuncs_.sigmoidGPU(input);
    }, 10);
    speedup = cpuTime / gpuTime;
    
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpuTime << " ms" << std::endl;
    std::cout << "  GPU: " << std::fixed << std::setprecision(2) << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    addResult("Sigmoid", "CPU", vectorSize, cpuTime);
    addResult("Sigmoid", "GPU", vectorSize, gpuTime, speedup);
    
    // Benchmark Tanh
    std::cout << "\nTanh:" << std::endl;
    cpuTime = measureTime([&]() {
        activationFuncs_.tanhCPU(input);
    }, 10);
    gpuTime = measureTime([&]() {
        activationFuncs_.tanhGPU(input);
    }, 10);
    speedup = cpuTime / gpuTime;
    
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpuTime << " ms" << std::endl;
    std::cout << "  GPU: " << std::fixed << std::setprecision(2) << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    addResult("Tanh", "CPU", vectorSize, cpuTime);
    addResult("Tanh", "GPU", vectorSize, gpuTime, speedup);
}

void Benchmark::benchmarkNeuralLayer(int inputSize, int hiddenSize, int outputSize) {
    std::cout << "\n=== Neural Network Layer Benchmark ===" << std::endl;
    std::cout << "Layer: " << inputSize << " -> " << hiddenSize << " -> " << outputSize << std::endl;
    
    // Create weights and input
    auto W1 = matrixOps_.createRandomMatrix(hiddenSize, inputSize, -0.1f, 0.1f);
    auto W2 = matrixOps_.createRandomMatrix(outputSize, hiddenSize, -0.1f, 0.1f);
    auto input = matrixOps_.createRandomMatrix(1, inputSize, -1.0f, 1.0f);
    
    // Benchmark forward pass on CPU
    double cpuTime = measureTime([&]() {
        // Input -> Hidden
        auto hidden = matrixOps_.multiplyCPU(input, W1, 1, inputSize, hiddenSize);
        // Apply ReLU
        for (auto& val : hidden) val = std::max(0.0f, val);
        // Hidden -> Output
        auto output = matrixOps_.multiplyCPU(hidden, W2, 1, hiddenSize, outputSize);
        // Apply Sigmoid
        for (auto& val : output) val = 1.0f / (1.0f + std::exp(-val));
    }, 5);
    
    // Benchmark forward pass on GPU
    double gpuTime = measureTime([&]() {
        // Input -> Hidden
        auto hidden = matrixOps_.multiplyGPU(input, W1, 1, inputSize, hiddenSize);
        // Apply ReLU
        hidden = activationFuncs_.reluGPU(hidden);
        // Hidden -> Output
        auto output = matrixOps_.multiplyGPU(hidden, W2, 1, hiddenSize, outputSize);
        // Apply Sigmoid
        output = activationFuncs_.sigmoidGPU(output);
    }, 5);
    
    double speedup = cpuTime / gpuTime;
    
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpuTime << " ms" << std::endl;
    std::cout << "  GPU: " << std::fixed << std::setprecision(2) << gpuTime << " ms" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    addResult("Neural Layer", "CPU", inputSize * hiddenSize * outputSize, cpuTime);
    addResult("Neural Layer", "GPU", inputSize * hiddenSize * outputSize, gpuTime, speedup);
}

template<typename Func>
double Benchmark::measureTime(Func func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0 / iterations; // Convert to milliseconds
}

void Benchmark::addResult(const std::string& operation, const std::string& implementation, 
                         int size, double time_ms, double speedup) {
    results_.push_back({operation, implementation, size, time_ms, speedup});
}

void Benchmark::printResults() const {
    std::cout << "\n=== Benchmark Summary ===" << std::endl;
    
    // Group results by operation
    std::vector<std::string> operations;
    for (const auto& result : results_) {
        if (std::find(operations.begin(), operations.end(), result.operation) == operations.end()) {
            operations.push_back(result.operation);
        }
    }
    
    for (const auto& operation : operations) {
        std::cout << "\n" << operation << ":" << std::endl;
        std::cout << std::setw(15) << "Size" << std::setw(15) << "CPU (ms)" 
                  << std::setw(15) << "GPU (ms)" << std::setw(15) << "Speedup" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        std::vector<BenchmarkResult> opResults;
        for (const auto& result : results_) {
            if (result.operation == operation) {
                opResults.push_back(result);
            }
        }
        
        // Sort by size
        std::sort(opResults.begin(), opResults.end(), 
                  [](const BenchmarkResult& a, const BenchmarkResult& b) {
                      return a.size < b.size;
                  });
        
        for (size_t i = 0; i < opResults.size(); i += 2) {
            if (i + 1 < opResults.size()) {
                const auto& cpu = opResults[i].implementation == "CPU" ? opResults[i] : opResults[i + 1];
                const auto& gpu = opResults[i].implementation == "GPU" ? opResults[i] : opResults[i + 1];
                
                std::cout << std::setw(15) << cpu.size 
                          << std::setw(15) << std::fixed << std::setprecision(2) << cpu.time_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << gpu.time_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << gpu.speedup << "x" << std::endl;
            }
        }
    }
}

void Benchmark::saveResultsToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "Operation,Implementation,Size,Time(ms),Speedup\n";
    for (const auto& result : results_) {
        file << result.operation << "," << result.implementation << "," 
             << result.size << "," << result.time_ms << "," << result.speedup << "\n";
    }
    
    std::cout << "Results saved to: " << filename << std::endl;
}
