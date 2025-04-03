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

__global__ void qkt_tiled_kernel(float* Q, float* K, float* S, int B, int N, int D, int H) {
    int D_H = D / H;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float* Q_h = Q + blockIdx.z * N * D_H;
    float* K_h = K + blockIdx.z * N * D_H;
    float* S_h = S + blockIdx.z * N * N;

    float sum = 0.0f;

    __shared__ float q_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float k_tile[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < (D_H + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int q_col = t * TILE_SIZE + threadIdx.x;
        int k_col = t * TILE_SIZE + threadIdx.y;

        if (row < N && q_col < D_H)
            q_tile[threadIdx.y][threadIdx.x] = Q_h[row * D_H + q_col];
        else
            q_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && k_col < D_H)
            k_tile[threadIdx.y][threadIdx.x] = K_h[col * D_H + k_col];
        else
            k_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += q_tile[threadIdx.y][i] * k_tile[i][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N) {
        S_h[row * N + col] = sum / sqrtf((float)D_H);
    }
}

__global__ void block_softmax_kernel(float* S, float* A, int N, int B, int H) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;

    int base = ((batch_id * H + head_id) * N + row) * N;

    float local_max = -1e20f;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = S[base + j];
        smem[j] = val;
        local_max = fmaxf(local_max, val);
    }

    __syncthreads();

    // Block-wide reduction to get global max
    __shared__ float max_val;
    if (tid == 0) {
        max_val = -1e20f;
        for (int j = 0; j < N; j++)
            max_val = fmaxf(max_val, smem[j]);
    }
    __syncthreads();

    // Compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        smem[j] = expf(smem[j] - max_val);
        local_sum += smem[j];
    }

    __syncthreads();

    // Block-wide reduction for sum
    __shared__ float total_sum;
    if (tid == 0) {
        total_sum = 0.0f;
        for (int j = 0; j < N; j++)
            total_sum += smem[j];
    }
    __syncthreads();

    // Normalize
    for (int j = tid; j < N; j += blockDim.x) {
        A[base + j] = smem[j] / total_sum;
    }
}

__global__ void av_tiled_kernel(float* A, float* V, float* O, int B, int N, int D, int H) {
    int D_H = D / H;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int head_idx = blockIdx.z % H;
    int batch_idx = blockIdx.z / H;

    float* A_h = A + (batch_idx * H + head_idx) * N * N;
    float* V_h = V + (batch_idx * H + head_idx) * N * D_H;
    float* O_h = O + (batch_idx * N + row) * D;

    float sum = 0.0f;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int v_row = t * TILE_SIZE + threadIdx.y;

        if (row < N && a_col < N)
            A_tile[threadIdx.y][threadIdx.x] = A_h[row * N + a_col];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (v_row < N && col < D_H)
            V_tile[threadIdx.y][threadIdx.x] = V_h[v_row * D_H + col];
        else
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[threadIdx.y][k] * V_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < D_H)
        O_h[head_idx * D_H + col] = sum;
}

void run_self_attention(int B, int N, int D, int H, float* h_Q, float* h_K, float* h_V, float* h_O) {
    int D_H = D / H;
    size_t qkv_size = B * H * N * D_H * sizeof(float);
    size_t smat_size = B * H * N * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_S, *d_A;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_S, smat_size);
    cudaMalloc(&d_A, smat_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_qkt, time_softmax, time_av;

    // QKT
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, B * H);
    cudaEventRecord(start);
    qkt_tiled_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_S, B, N, D, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_qkt, start, stop);

    // Softmax
    dim3 smBlockDim(256);
    dim3 smGridDim(N, H, B);
    size_t smem_size = N * sizeof(float);
    cudaEventRecord(start);
    block_softmax_kernel<<<smGridDim, smBlockDim, smem_size>>>(d_S, d_A, N, B, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_softmax, start, stop);

    // AV
    dim3 avBlockDim(TILE_SIZE, TILE_SIZE);
    dim3 avGridDim((D_H + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, B * H);
    cudaEventRecord(start);
    av_tiled_kernel<<<avGridDim, avBlockDim>>>(d_A, d_V, d_O, B, N, D, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_av, start, stop);

    cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    printf("Execution Time (ms): QKT = %.2fms, Softmax = %.2fms, AV = %.2fms\n", time_qkt, time_softmax, time_av);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_S);
    cudaFree(d_A);
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
    for (int i = 0; i < B * H * N * D_H; i++) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    run_self_attention(B, N, D, H, h_Q, h_K, h_V, h_O);

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
