#include "vertex_processor.h"
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void scaleVerticesKernel(float* vertices, int vertex_count, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < vertex_count * 3) {
        vertices[idx] *= scale_factor;
    }
}

__global__ void scaleVerticesKernelDouble(double* vertices, int vertex_count, float scale_factor) {
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

__global__ void translateVerticesKernelDouble(double* vertices, int vertex_count, float dx, float dy, float dz) {
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

__global__ void smooth_by_edges(double* V, int* E, int edge_count, double* V_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < edge_count) {
      int first  = E[(idx*2)] * 3;
      int second = E[(idx*2)+1] * 3;

      atomicAdd(&V_out[first + 0], V[second + 0]);
      atomicAdd(&V_out[first + 1], V[second + 1]);
      atomicAdd(&V_out[first + 2], V[second + 2]);

      atomicAdd(&V_out[second + 0], V[first + 0]);
      atomicAdd(&V_out[second + 1], V[first + 1]);
      atomicAdd(&V_out[second + 2], V[first + 2]);
    }
}

__global__ void smooth_by_edges2(double* V, int* E, int *prefix_sum, int edge_count, double* V_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < edge_count) {
      int first  = E[(idx*2)];
      int second = E[(idx*2)+1];

      double sum1 = 1 / (double) (first == 0 ? prefix_sum[first] : prefix_sum[first] - prefix_sum[first-1]);
      double sum2 = 1 / (double) (second == 0 ? prefix_sum[second] : prefix_sum[second] - prefix_sum[second-1]);

      first  *= 3;
      second *= 3;

      atomicAdd(&V_out[first + 0], V[second + 0] * (sum1));
      atomicAdd(&V_out[first + 1], V[second + 1] * (sum1));
      atomicAdd(&V_out[first + 2], V[second + 2] * (sum1));

      atomicAdd(&V_out[second + 0], V[first + 0] * (sum2));
      atomicAdd(&V_out[second + 1], V[first + 1] * (sum2));
      atomicAdd(&V_out[second + 2], V[first + 2] * (sum2));
    }
}

__global__ void average_by_sum(double* V, int* prefix_sum, int vert_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < vert_count) {
      int sum = idx == 0 ? prefix_sum[0] : prefix_sum[idx] - prefix_sum[idx-1];
      V[idx * 3 + 0] = V[idx * 3 + 0] / ((double) sum);
      V[idx * 3 + 1] = V[idx * 3 + 1] / ((double) sum);
      V[idx * 3 + 2] = V[idx * 3 + 2] / ((double) sum);
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

void processVerticesFromPointer(double* vertices, int vertex_count, float scale_factor) {
    double* d_vertices;
    size_t size = vertex_count * 3 * sizeof(double);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices, size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(scaleVerticesKernelDouble, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, scale_factor);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices, d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Scaled " << vertex_count << " vertices by factor " << scale_factor << std::endl;
}

void translateVerticesFromPointer(double* vertices, int vertex_count, float dx, float dy, float dz) {
    double* d_vertices;
    size_t size = vertex_count * 3 * sizeof(double);
    
    hipMalloc(&d_vertices, size);
    hipMemcpy(d_vertices, vertices, size, hipMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertex_count * 3 + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(translateVerticesKernelDouble, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_vertices, vertex_count, dx, dy, dz);
    
    hipDeviceSynchronize();
    hipMemcpy(vertices, d_vertices, size, hipMemcpyDeviceToHost);
    hipFree(d_vertices);
    
    std::cout << "GPU: Translated " << vertex_count << " vertices by (" << dx << ", " << dy << ", " << dz << ")" << std::endl;
}

void smooth_tets_naive(double* TV, int vertex_count, int* edge_pairs, int num_edges, int* prefix_sum) {
    double* d_V;
    double* d_V_out;
    int* d_E;
    int* d_prefix_sum;

    size_t size_verts = vertex_count * 3 * sizeof(double);
    size_t size_edges = num_edges * 2 * sizeof(int);
    size_t size_prefix_sum = vertex_count * sizeof(int);
    
    hipMalloc(&d_V, size_verts);
    hipMalloc(&d_V_out, size_verts);
    hipMalloc(&d_E, size_edges);
    hipMalloc(&d_prefix_sum, size_prefix_sum);

    hipMemcpy(d_V, TV, size_verts, hipMemcpyHostToDevice);
    hipMemcpy(d_E, edge_pairs, size_edges, hipMemcpyHostToDevice);
    hipMemcpy(d_prefix_sum, prefix_sum, size_prefix_sum, hipMemcpyHostToDevice);
    hipMemset(d_V_out, 0, size_verts);
    
    int threadsPerBlock = 256;
    int blocksPerGridEdges = (num_edges + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << blocksPerGridEdges << std::endl;

    int blocksPerGridVerts = (vertex_count + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << blocksPerGridVerts << std::endl;
    //hipLaunchKernelGGL(smooth_by_edges, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0, d_V, d_E, num_edges, d_V_out);
    //hipDeviceSynchronize();

    hipLaunchKernelGGL(smooth_by_edges2, dim3(blocksPerGridEdges), dim3(threadsPerBlock), 0, 0, d_V, d_E, d_prefix_sum, num_edges, d_V_out);
    hipDeviceSynchronize();
    

    //hipLaunchKernelGGL(average_by_sum, dim3(blocksPerGridVerts), dim3(threadsPerBlock), 0, 0, d_V_out, d_prefix_sum, vertex_count);
    //hipDeviceSynchronize();

    hipMemcpy(TV, d_V_out, size_verts, hipMemcpyDeviceToHost);

    hipFree(d_V);
    hipFree(d_V_out);
    hipFree(d_E);
    hipFree(d_prefix_sum);
}

void printGPUInfo() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    
    for (int device = 0; device < deviceCount; device++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device);
        
        std::cout << "\nDevice " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimensions: (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Memory pitch: " << prop.memPitch / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Texture alignment: " << prop.textureAlignment << " bytes" << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Kernel execution timeout: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Integrated GPU: " << (prop.integrated ? "Yes" : "No") << std::endl;
        std::cout << "  Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        
        // Get current memory usage
        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);
        std::cout << "  Memory usage: " << (total_mem - free_mem) / (1024 * 1024) 
                  << " MB used / " << total_mem / (1024 * 1024) << " MB total" << std::endl;
    }
    
    // Example kernel launch configuration info
    std::cout << "\n=== Example Kernel Launch Configuration ===" << std::endl;
    int example_data_size = 10000;
    int threadsPerBlock = 256;
    int blocksPerGrid = (example_data_size + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "For " << example_data_size << " elements:" << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  Blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "  Total threads: " << blocksPerGrid * threadsPerBlock << std::endl;
    std::cout << "  Thread utilization: " << (float)example_data_size / (blocksPerGrid * threadsPerBlock) * 100 << "%" << std::endl;
}

}
