#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

void cpu_self_attention(float* Q, float* K, float* V, float* O, int B, int N, int D, int H) {
    int D_H = D / H;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            float* Q_h = Q + (b * H + h) * N * D_H;
            float* K_h = K + (b * H + h) * N * D_H;
            float* V_h = V + (b * H + h) * N * D_H;

            float* S = (float*)malloc(N * N * sizeof(float));
            float* A = (float*)malloc(N * N * sizeof(float));

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < D_H; ++k) {
                        sum += Q_h[i * D_H + k] * K_h[j * D_H + k];
                    }
                    S[i * N + j] = sum / sqrtf((float)D_H);
                }
            }

            for (int i = 0; i < N; ++i) {
                float max_val = -1e20f;
                for (int j = 0; j < N; ++j)
                    max_val = fmaxf(max_val, S[i * N + j]);

                float sum_exp = 0.0f;
                for (int j = 0; j < N; ++j) {
                    A[i * N + j] = expf(S[i * N + j] - max_val);
                    sum_exp += A[i * N + j];
                }
                for (int j = 0; j < N; ++j)
                    A[i * N + j] /= sum_exp;
            }

            for (int i = 0; i < N; ++i) {
                for (int d = 0; d < D_H; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        sum += A[i * N + j] * V_h[j * D_H + d];
                    }
                    O[b * N * D + i * D + h * D_H + d] = sum;
                }
            }

            free(S); free(A);
        }
    }
}

// QK^T kernel with head parallelized
__global__ void kernel_qk_dot_multi(float* Q, float* K, float* S, int B, int H, int N, int D_H) {
    int bh = blockIdx.z;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < D_H; ++k) {
            float q = Q[((b * H + h) * N + i) * D_H + k];
            float k_val = K[((b * H + h) * N + j) * D_H + k];
            sum += q * k_val;
        }
        S[((b * H + h) * N + i) * N + j] = sum / sqrtf((float)D_H);
    }
}

// Softmax kernel with head parallelized
__global__ void kernel_softmax_multi(float* S, float* A, int B, int H, int N) {
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float max_val = -1e9f;
        for (int j = 0; j < N; ++j)
            max_val = fmaxf(max_val, S[((b * H + h) * N + i) * N + j]);
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            A[((b * H + h) * N + i) * N + j] = expf(S[((b * H + h) * N + i) * N + j] - max_val);
            sum_exp += A[((b * H + h) * N + i) * N + j];
        }
        for (int j = 0; j < N; ++j)
            A[((b * H + h) * N + i) * N + j] /= sum_exp;
    }
}

// AV kernel with concat output and head parallelized
__global__ void kernel_av_concat_multi(float* A, float* V, float* O_concat, int B, int H, int N, int D_H, int D) {
    int bh = blockIdx.z;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && d < D_H) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
            sum += A[((b * H + h) * N + i) * N + j] * V[((b * H + h) * N + j) * D_H + d];
        O_concat[(b * N + i) * D + h * D_H + d] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <Batch Size (B)> <Rows (N)> <Columns (D)> <Heads (H)> <Verify (true or false)>\n", argv[0]);
        return 1;
    }

    int B = atoi(argv[1]);
    int N = atoi(argv[2]);
    int D = atoi(argv[3]);
    int H = atoi(argv[4]);
    int verify = (strcmp(argv[5], "true") == 0);

    if (D % H != 0) {
        fprintf(stderr, "Error: D must be divisible by H.\n");
        return 1;
    }

    printf("Running Self-Attention on GPU (Fully Sequential) with Parameters:\n");
    printf("Batch Size (B): %d, Rows (N): %d, Columns (D): %d, Heads (H): %d\n", B, N, D, H);
    printf("Verification: %s\n", verify ? "Enabled" : "Disabled");

    int D_H = D / H;
    size_t qkv_size = B * H * N * D_H * sizeof(float);
    size_t score_size = B * H * N * N * sizeof(float);
    size_t output_size = B * N * D * sizeof(float);

    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_O = (float*)malloc(output_size);

    for (int i = 0; i < B * H * N * D_H; ++i) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_S, *d_A, *d_O;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_S, score_size);
    cudaMalloc(&d_A, score_size);
    cudaMalloc(&d_O, output_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_qkt, time_softmax, time_av;

    // QKT
    dim3 block_qk(TILE_SIZE, TILE_SIZE);
    dim3 grid_qk((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE, B * H);
    cudaEventRecord(start);
    kernel_qk_dot_multi<<<grid_qk, block_qk>>>(d_Q, d_K, d_S, B, H, N, D_H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_qkt, start, stop);

    // Softmax
    dim3 block_softmax(256);
    dim3 grid_softmax((N + 255)/256, B * H);
    cudaEventRecord(start);
    kernel_softmax_multi<<<grid_softmax, block_softmax>>>(d_S, d_A, B, H, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_softmax, start, stop);

    // AV
    dim3 block_av(TILE_SIZE, TILE_SIZE);
    dim3 grid_av((D_H + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE, B * H);
    cudaEventRecord(start);
    kernel_av_concat_multi<<<grid_av, block_av>>>(d_A, d_V, d_O, B, H, N, D_H, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_av, start, stop);

    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    printf("Execution Time (ms): QKT = %.2fms, Softmax = %.2fms, AV = %.2fms\n", time_qkt, time_softmax, time_av);

    float *h_O_cpu = NULL;
    if (verify) {
        h_O_cpu = (float*)malloc(output_size);
        cpu_self_attention(h_Q, h_K, h_V, h_O_cpu, B, N, D, H);

        float max_diff = 0.0f;
        for (int i = 0; i < B * N * D; ++i) {
            float diff = fabs(h_O[i] - h_O_cpu[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max absolute error between GPU and CPU: %.6f\n", max_diff);
        free(h_O_cpu);
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_A); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O);
    return 0;
}

