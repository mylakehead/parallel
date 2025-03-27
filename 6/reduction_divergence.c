#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define REPEAT 1000

__global__ void reduceKernel(float* input, float* output) {
    __shared__ float partialSum[N];
    unsigned int t = threadIdx.x;

    partialSum[t] = input[t];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0)
            partialSum[t] += partialSum[t + stride];
    }

    if (t == 0) output[0] = partialSum[0];
}

int main() {
    float h_input[N], h_output;
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < REPEAT; i++) {
        reduceKernel<<<1, N>>>(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum: %f\n", h_output);
    printf("total time for %d runs %f ms\n", REPEAT, ms);
    printf("average cost: %f ms\n", ms/REPEAT);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}