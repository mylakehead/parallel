#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


// CUDA Kernel for sequential self-attention
__global__ void sequential_self_attention_kernel(float* Q, float* K, float* V, float* O, int B, int N, int D, int H) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        // dimension per head
        int D_H = D / H;

        for (int sample_idx = 0; sample_idx < B; sample_idx++) {
            float* concat_output = (float*)malloc(N * D * sizeof(float));

            // Q[B][H][N][D_H]
            for (int head = 0; head < H; head++) {
                float* Q_h = Q + sample_idx * N * D + head * N * D_H;
                float* K_h = K + sample_idx * N * D + head * N * D_H;
                float* V_h = V + sample_idx * N * D + head * N * D_H;
                float* O_h = O + sample_idx * N * D + head * N * D_H;

                // Step 1: Compute QK^T
                float* S = (float*)malloc(N * N * sizeof(float));

                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0.0f;
                        for (int k = 0; k < D_H; k++) {
                            sum += Q_h[i * D_H + k] * K_h[j * D_H + k];
                        }
                        S[i * N + j] = sum / sqrtf(D_H);
                    }
                }

                // Step 2: Apply Softmax
                float* A = (float*)malloc(N * N * sizeof(float));

                for (int i = 0; i < N; i++) {
                    float row_max = -1e9;
                    for (int j = 0; j < N; j++) {
                        if (S[i * N + j] > row_max) row_max = S[i * N + j];
                    }

                    float sum_exp = 0.0f;
                    for (int j = 0; j < N; j++) {
                        A[i * N + j] = expf(S[i * N + j] - row_max);
                        sum_exp += A[i * N + j];
                    }

                    for (int j = 0; j < N; j++) {
                        A[i * N + j] /= sum_exp;
                    }
                }

                // Step 3: Compute AV
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < D_H; k++) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; j++) {
                            sum += A[i * N + j] * V_h[j * D_H + k];
                        }
                        O_h[i * D_H + k] = sum;

                        // Step 4: Concatenation (Each head's output is placed in the correct segment)
                        concat_output[i * D + head * D_H + k] = sum;
                    }
                }

                free(S);
                free(A);
            }

            // Copy concatenated output to final output matrix
            for (int i = 0; i < N * D; i++) {
                O[sample_idx * N * D + i] = concat_output[i];
            }

            free(concat_output);
        }
    }
}

// Host function to run Self-Attention on GPU (Sequential Execution)
void run_self_attention(int B, int N, int D, int H) {
    size_t size = B * N * D * sizeof(float);

    // Allocate memory for input and output matrices
    float* h_Q = (float*)malloc(size);
    float* h_K = (float*)malloc(size);
    float* h_V = (float*)malloc(size);
    float* h_O = (float*)malloc(size);

    // Initialize input matrices with random values
    for (size_t i = 0; i < B * N * D; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    // Allocate GPU memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void**)&d_Q, size);
    cudaMalloc((void**)&d_K, size);
    cudaMalloc((void**)&d_V, size);
    cudaMalloc((void**)&d_O, size);

    // Copy data to GPU
    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    // CUDA Event Timing
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start, 0);

    // Launch kernel: Only one block, one thread executes everything sequentially
    sequential_self_attention_kernel<<<1, 1>>>(d_Q, d_K, d_V, d_O, B, N, D, H);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy result back to CPU
    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    printf("Self-Attention Computation Completed on GPU (Sequential Execution)!\n");
    printf("GPU Execution Time: %.4f ms\n", elapsedTime);

    // Free GPU memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    // Free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <Batch Size (B)> <Rows (N)> <Columns (D)> <Heads (H)>\n", argv[0]);
        return 1;
    }

    int B = atoi(argv[1]);  // Batch size (number of samples)
    int N = atoi(argv[2]);  // Number of rows (tokens per sample)
    int D = atoi(argv[3]);  // Feature dimension per token
    int H = atoi(argv[4]);  // Number of attention heads

    if (D % H != 0) {
        fprintf(stderr, "Error: Feature dimension (D) must be divisible by the number of heads (H).\n");
        return 1;
    }

    printf("Running Self-Attention on GPU (Fully Sequential) with Parameters:\n");
    printf("Batch Size (B): %d, Rows (N): %d, Columns (D): %d, Heads (H): %d\n", N, D, H, B);

    run_self_attention(B, N, D, H);
    return 0;
}