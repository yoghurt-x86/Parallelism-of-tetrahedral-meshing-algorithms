#include "vertex_processor.h"
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void scaleVerticesKernel(float* vertices, int vertex_count, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vertex_count * 3) {
        vertices[idx] *= scale_factor;
    }
}

__global__ void translateVerticesKernel(float* vertices, int vertex_count, float dx, float dy, float dz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vertex_idx = idx / 3;
    int coord_idx = idx % 3;
    
    if (vertex_idx < vertex_count) {
        if (coord_idx == 0) {
            vertices[idx] += dx;
        } else if (coord_idx == 1) {
            vertices[idx] += dy;
        } else if (coord_idx == 2) {
            vertices[idx] += dz;
        }
    }
}

namespace VertexProcessor {

void processVertices(std::vector<float>& vertices, int vertex_count, float scale_factor) {
    float* d_vertices;
    size_t size = vertex_count * 3 * sizeof(float);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices.data(), size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(scaleVerticesKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, scale_factor);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices.data(), d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Scaled " << vertex_count << " vertices by factor " << scale_factor << std::endl;
}

void translateVertices(std::vector<float>& vertices, int vertex_count, float dx, float dy, float dz) {
    float* d_vertices;
    size_t size = vertex_count * 3 * sizeof(float);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices.data(), size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(translateVerticesKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, dx, dy, dz);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices.data(), d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Translated " << vertex_count << " vertices by (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
}

}