cmake_minimum_required(VERSION 3.16)
project(OpenCL_Neural_Network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Find OpenMP for CPU benchmarking
find_package(OpenMP)

# Source files
set(SOURCES
    src/main.cpp
    src/opencl_utils.cpp
    src/matrix_ops.cpp
    src/activation_functions.cpp
    src/benchmark.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
if(APPLE)
    target_include_directories(${PROJECT_NAME} PRIVATE
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers
        include
    )
else()
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCL_INCLUDE_DIRS}
        include
    )
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O3
)
