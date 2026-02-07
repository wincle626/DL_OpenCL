#include "opencl_utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

OpenCLContext::OpenCLContext() {
    selectPlatform();
    selectDevice();
    createContext();
    createCommandQueue();
}

OpenCLContext::~OpenCLContext() {
    if (commandQueue_) {
        clReleaseCommandQueue(commandQueue_);
    }
    if (context_) {
        clReleaseContext(context_);
    }
}

void OpenCLContext::selectPlatform() {
    cl_uint numPlatforms;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    checkOpenCLError(error, "Getting number of platforms");
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    checkOpenCLError(error, "Getting platform IDs");
    
    // Select first available platform (can be enhanced to select specific ones)
    platform_ = platforms[0];
    
    std::cout << "Selected platform: ";
    char platformName[256];
    error = clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    if (error == CL_SUCCESS) {
        std::cout << platformName << std::endl;
    }
}

void OpenCLContext::selectDevice() {
    cl_uint numDevices;
    cl_int error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    
    if (error != CL_SUCCESS || numDevices == 0) {
        // Fallback to CPU if no GPU available
        std::cout << "No GPU found, falling back to CPU" << std::endl;
        error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr, &numDevices);
        checkOpenCLError(error, "Getting number of CPU devices");
    }
    
    std::vector<cl_device_id> devices(numDevices);
    error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (error != CL_SUCCESS) {
        error = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, numDevices, devices.data(), nullptr);
        checkOpenCLError(error, "Getting device IDs");
    }
    
    // Select first available device
    device_ = devices[0];
    
    std::cout << "Selected device: ";
    char deviceName[256];
    error = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    if (error == CL_SUCCESS) {
        std::cout << deviceName << std::endl;
    }
}

void OpenCLContext::createContext() {
    cl_int error;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
    checkOpenCLError(error, "Creating context");
}

void OpenCLContext::createCommandQueue() {
    cl_int error;
    commandQueue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &error);
    checkOpenCLError(error, "Creating command queue");
}

void OpenCLContext::printDeviceInfo() const {
    cl_ulong globalMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxWorkGroupSize;
    
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    
    std::cout << "Device Info:" << std::endl;
    std::cout << "  Global Memory: " << globalMemSize / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Compute Units: " << maxComputeUnits << std::endl;
    std::cout << "  Max Work Group Size: " << maxWorkGroupSize << std::endl;
}

OpenCLKernel::OpenCLKernel(const OpenCLContext& context, const std::string& source, const std::string& kernelName) {
    compileProgram(context, source);
    createKernel(kernelName);
}

OpenCLKernel::~OpenCLKernel() {
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
    if (program_) {
        clReleaseProgram(program_);
    }
}

void OpenCLKernel::compileProgram(const OpenCLContext& context, const std::string& source) {
    cl_int error;
    const char* sourcePtr = source.c_str();
    size_t sourceSize = source.length();
    
    program_ = clCreateProgramWithSource(context.getContext(), 1, &sourcePtr, &sourceSize, &error);
    checkOpenCLError(error, "Creating program");
    
    cl_device_id device = context.getDevice();
    error = clBuildProgram(program_, 1, &device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program_, context.getDevice(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program_, context.getDevice(), CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log: " << log.data() << std::endl;
        checkOpenCLError(error, "Building program");
    }
}

void OpenCLKernel::createKernel(const std::string& kernelName) {
    cl_int error;
    kernel_ = clCreateKernel(program_, kernelName.c_str(), &error);
    checkOpenCLError(error, "Creating kernel");
}

OpenCLBuffer::OpenCLBuffer(const OpenCLContext& context, size_t size, cl_mem_flags flags)
    : size_(size), context_(context) {
    cl_int error;
    buffer_ = clCreateBuffer(context_.getContext(), flags, size, nullptr, &error);
    checkOpenCLError(error, "Creating buffer");
}

OpenCLBuffer::~OpenCLBuffer() {
    if (buffer_) {
        clReleaseMemObject(buffer_);
    }
}

void OpenCLBuffer::writeData(const void* data, size_t offset) {
    cl_int error = clEnqueueWriteBuffer(context_.getCommandQueue(), buffer_, CL_TRUE, offset, 
                                       size_ - offset, data, 0, nullptr, nullptr);
    checkOpenCLError(error, "Writing buffer data");
}

void OpenCLBuffer::readData(void* data, size_t offset) {
    cl_int error = clEnqueueReadBuffer(context_.getCommandQueue(), buffer_, CL_TRUE, offset, 
                                      size_ - offset, data, 0, nullptr, nullptr);
    checkOpenCLError(error, "Reading buffer data");
}

std::string readKernelSource(const std::string& filename) {
    // Try multiple possible paths for kernel files
    std::vector<std::string> searchPaths = {
        filename,  // Current directory
        "../" + filename,  // Parent directory
        "../../" + filename,  // Two levels up
        "kernels/" + filename.substr(filename.find_last_of('/') + 1),  // kernels subdirectory
        "../kernels/" + filename.substr(filename.find_last_of('/') + 1)  // kernels from parent
    };
    
    for (const auto& path : searchPaths) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }
    }
    
    throw std::runtime_error("Could not open kernel file: " + filename + " (tried multiple paths)");
}

void checkOpenCLError(cl_int error, const std::string& operation) {
    if (error != CL_SUCCESS) {
        std::string errorMsg = "OpenCL error in " + operation + ": " + std::to_string(error);
        throw std::runtime_error(errorMsg);
    }
}
