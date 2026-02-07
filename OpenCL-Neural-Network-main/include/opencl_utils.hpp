#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

class OpenCLContext {
public:
    OpenCLContext();
    ~OpenCLContext();
    
    cl_context getContext() const { return context_; }
    cl_command_queue getCommandQueue() const { return commandQueue_; }
    cl_device_id getDevice() const { return device_; }
    
    void printDeviceInfo() const;
    
private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue commandQueue_;
    
    void selectPlatform();
    void selectDevice();
    void createContext();
    void createCommandQueue();
};

class OpenCLKernel {
public:
    OpenCLKernel(const OpenCLContext& context, const std::string& source, const std::string& kernelName);
    ~OpenCLKernel();
    
    cl_kernel getKernel() const { return kernel_; }
    cl_program getProgram() const { return program_; }
    
private:
    cl_program program_;
    cl_kernel kernel_;
    
    void compileProgram(const OpenCLContext& context, const std::string& source);
    void createKernel(const std::string& kernelName);
};

class OpenCLBuffer {
public:
    OpenCLBuffer(const OpenCLContext& context, size_t size, cl_mem_flags flags);
    ~OpenCLBuffer();
    
    cl_mem getBuffer() const { return buffer_; }
    size_t getSize() const { return size_; }
    
    void writeData(const void* data, size_t offset = 0);
    void readData(void* data, size_t offset = 0);
    
private:
    cl_mem buffer_;
    size_t size_;
    const OpenCLContext& context_;
};

// Utility functions
std::string readKernelSource(const std::string& filename);
void checkOpenCLError(cl_int error, const std::string& operation);
