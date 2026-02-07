#!/bin/bash

# Build script for OpenCL Neural Network Primitives

echo "Building OpenCL Neural Network Primitives..."

# Check if cmake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake is not installed. Please install cmake first."
    exit 1
fi

# Check if make is available
if ! command -v make &> /dev/null; then
    echo "Error: make is not installed. Please install make first."
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with cmake
echo "Configuring with cmake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "Error: cmake configuration failed."
    exit 1
fi

# Build the project
echo "Building project..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: build failed."
    exit 1
fi

echo "Build completed successfully!"
echo "Run the application with: ./build/OpenCL_Neural_Network"
