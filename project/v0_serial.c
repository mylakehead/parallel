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

__global__ void sequential_self_attention_kernel(float* Q, float* K, float* V, float* O, int B, int N, int D, int H) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        int D_H = D / H;

        for (int b = 0; b < B; ++b) {
            float* concat_output = (float*)malloc(N * D * sizeof(float));

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
                    float row_max = -1e9;
                    for (int j = 0; j < N; ++j)
                        if (S[i * N + j] > row_max) row_max = S[i * N + j];

                    float sum_exp = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        A[i * N + j] = expf(S[i * N + j] - row_max);
                        sum_exp += A[i * N + j];
                    }

                    for (int j = 0; j < N; ++j)
                        A[i * N + j] /= sum_exp;
                }

                for (int i = 0; i < N; ++i) {
                    for (int k = 0; k < D_H; ++k) {
                        float sum = 0.0f;
                        for (int j = 0; j < N; ++j) {
                            sum += A[i * N + j] * V_h[j * D_H + k];
                        }
                        concat_output[i * D + h * D_H + k] = sum;
                    }
                }

                free(S); free(A);
            }

            for (int i = 0; i < N * D; i++) {
                O[b * N * D + i] = concat_output[i];
            }

            free(concat_output);
        }
    }
}

void run_self_attention(float* h_Q, float* h_K, float* h_V, float* h_O, int B, int N, int D, int H) {
    int D_H = D / H;
    size_t qkv_size = B * H * N * D_H * sizeof(float);
    size_t output_size = B * N * D * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void**)&d_Q, qkv_size);
    cudaMalloc((void**)&d_K, qkv_size);
    cudaMalloc((void**)&d_V, qkv_size);
    cudaMalloc((void**)&d_O, output_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sequential_self_attention_kernel<<<1, 1>>>(d_Q, d_K, d_V, d_O, B, N, D, H);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    printf("Self-Attention Computation Completed on GPU (Sequential Execution)!\n");
    printf("GPU Execution Time: %.4f ms\n", elapsedTime);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
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
    size_t output_size = B * N * D * sizeof(float);

    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_O = (float*)malloc(output_size);

    for (int i = 0; i < B * H * N * D_H; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    run_self_attention(h_Q, h_K, h_V, h_O, B, N, D, H);

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

    free(h_Q); free(h_K); free(h_V); free(h_O);
    return 0;
}