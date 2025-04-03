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

__global__ void fused_attention_tile_kernel(float* Q, float* K, float* V, float* O, int B, int H, int N, int D_H, int D) {
    int tile_col = threadIdx.x;
    int tile_row = threadIdx.y;
    int d_offset = blockIdx.x * TILE_SIZE + tile_col;
    int n_offset = blockIdx.y * TILE_SIZE + tile_row;
    int bh = blockIdx.z;
    int b = bh / H;
    int h = bh % H;

    if (d_offset >= D_H || n_offset >= N) return;

    // Offsets
    float* Q_h = Q + ((b * H + h) * N * D_H);
    float* K_h = K + ((b * H + h) * N * D_H);
    float* V_h = V + ((b * H + h) * N * D_H);
    float* O_h = O + ((b * N + n_offset) * D + h * D_H);

    // Step 1: Compute attention scores for n_offset-th row
    float max_score = -1e20f;
    float scores[1024]; // N max = 512
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < D_H; ++k) {
            dot += Q_h[n_offset * D_H + k] * K_h[j * D_H + k];
        }
        dot /= sqrtf((float)D_H);
        scores[j] = dot;
        max_score = fmaxf(max_score, dot);
    }

    // Step 2: Softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }

    // Step 3: Weighted sum for output[n_offset][d_offset]
    float out = 0.0f;
    for (int j = 0; j < N; ++j) {
        float weight = scores[j] / sum_exp;
        out += weight * V_h[j * D_H + d_offset];
    }

    O_h[d_offset] = out;
}

void run_fused_attention(float* h_Q, float* h_K, float* h_V, float* h_O, int B, int N, int D, int H) {
    int D_H = D / H;
    size_t qkv_size = B * H * N * D_H * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((D_H + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, B * H);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fused_attention_tile_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, B, H, N, D_H, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("[✓] Tile-Based Fused Attention Time: %.3f ms\n", elapsed_ms);

    cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
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

    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_O = (float*)malloc(qkv_size);
    for (int i = 0; i < B * H * N * D_H; ++i) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    run_fused_attention(h_Q, h_K, h_V, h_O, B, N, D, H);

    float *h_O_cpu = NULL;
    if (verify) {
        h_O_cpu = (float*)malloc(qkv_size);
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

