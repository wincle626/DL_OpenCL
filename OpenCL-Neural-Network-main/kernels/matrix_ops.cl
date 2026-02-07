// Matrix multiplication kernel: C = A * B
// Global work size: (colsB, rowsA)
// Local work size: (16, 16) for optimal performance
__kernel void matrix_multiply(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int rowsA,
    const int colsA,
    const int colsB
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Matrix addition kernel: C = A + B
__kernel void matrix_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size
) {
    int idx = get_global_id(0);
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

// Matrix transpose kernel: B = A^T
__kernel void matrix_transpose(
    __global const float* A,
    __global float* B,
    const int rows,
    const int cols
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// Optimized matrix multiplication using local memory
__kernel void matrix_multiply_optimized(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int rowsA,
    const int colsA,
    const int colsB
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    __local float localA[16][16];
    __local float localB[16][16];
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (colsA + 15) / 16; tile++) {
        int localRow = get_local_id(1);
        int localCol = get_local_id(0);
        
        // Load tiles into local memory
        if (row < rowsA && tile * 16 + localCol < colsA) {
            localA[localRow][localCol] = A[row * colsA + tile * 16 + localCol];
        } else {
            localA[localRow][localCol] = 0.0f;
        }
        
        if (col < colsB && tile * 16 + localRow < colsA) {
            localB[localRow][localCol] = B[(tile * 16 + localRow) * colsB + col];
        } else {
            localB[localRow][localCol] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum
        for (int k = 0; k < 16; k++) {
            sum += localA[localRow][k] * localB[k][localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}
