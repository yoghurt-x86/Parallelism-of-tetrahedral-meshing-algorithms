#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>

#define CPU_RUNS 1
#define GPU_RUNS 300

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// Copied from HelperCode/Lab-1-Cuda/helper.h
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


void calc(float *X, float *Y){
    const float temp = *X / (*X - 2.3 );

    *Y = temp * temp * temp;
}

void runOnCPU(float* X, float *Y, unsigned int N) {
    for(unsigned int i=0; i<N; i++) {
        calc(&X[i], &Y[i]);
    }
}

__global__ void myKernel(float* X, float *Y, unsigned int N) {
    //const unsigned int gid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < N) {
        // \ x → (x/(x-2.3))^3
        const float temp = X[gid] / (X[gid] - 2.3 );
        Y[gid] = temp * temp * temp;
    }
}


int main(int argc, char** argv) {
    unsigned long long N;

    { // reading the number of elements 
      if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      }

      N = atoll(argv[1]);
      printf("N is: %llu\n", N);

      const unsigned int maxN = 4294967295;

      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }


    unsigned long long mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);

    float* h_cpu_out = (float*) malloc(mem_size);
    float* h_gpu_out = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)i;
    }

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        runOnCPU(h_in, h_cpu_out, N);
    }

    double cpu_elapsed;
    {
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < CPU_RUNS; r++) {
            runOnCPU(h_in, h_cpu_out, N);
        }

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        cpu_elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / CPU_RUNS;
    }


    // allocate device memory
    float* d_in;
    float* d_out;

    unsigned int B = 256;
    unsigned int numblocks = (N + B - 1) / B;
    dim3 block (B, 1, 1);
    dim3 grid (numblocks, 1, 1);

    // use the first CUDA device:
    cudaSetDevice(0);
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        myKernel<<<grid, block>>>(d_in, d_out, N);
    }
  

    double gpu_elapsed;
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            myKernel<<<grid, block>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        gpu_elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
    }
        
    // check for errors
    //gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_gpu_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    for(unsigned int i=0; i<N; ++i) {
        float cpu_result = h_cpu_out[i];
        float gpu_result = h_gpu_out[i];


        if (fabs(cpu_result - gpu_result) >= 0.0001) {
            printf("INVALID\n");
            printf("Invalid result at index %d, cpu: %f, gpu: %f. \n", i, cpu_result, gpu_result);
            exit(3);
        }
    }

    double cpu_gigabytespersec = (2.0 * N * 4.0) / (cpu_elapsed * 1000.0);
    double gpu_gigabytespersec = (2.0 * N * 4.0) / (gpu_elapsed * 1000.0);
    printf("VALID\n");
    printf("The CPU took on average %f microseconds. GB/sec: %f \n", cpu_elapsed, cpu_gigabytespersec);
    printf("The GPU kernel took on average %f microseconds. GB/sec: %f \n", gpu_elapsed, gpu_gigabytespersec);
    printf("Acceleration: %.2fx\n", cpu_elapsed / gpu_elapsed);

    // clean-up memory
    free(h_in);       free(h_cpu_out);   free(h_gpu_out);
    cudaFree(d_in);   cudaFree(d_out);
}
