#include <stdio.h>
#include <cuda_runtime.h>

#define REPEAT 1000
#define N_WIDTH 8
#define MASK_WIDTH 3
#define BLOCK_WIDTH 4
#define TILE_WIDTH (BLOCK_WIDTH + 2 * (MASK_WIDTH / 2))

__constant__ float d_M[MASK_WIDTH * MASK_WIDTH];

__global__ void convolution(float *input, float *output, int n_width, int m_width) {
    int row_o = blockIdx.y * blockDim.y + threadIdx.y;
    int col_o = blockIdx.x * blockDim.x + threadIdx.x;

    int n = m_width / 2;

    float p_value = 0.0f;
    if (row_o < n_width && col_o < n_width) {
        for (int i = -n; i <= n; i++) {
            for (int j = -n; j <= n; j++) {
                int row_i = row_o + i;
                int col_i = col_o + j;

                float input_value = 0.0f;
                if (row_i >= 0 && row_i < n_width && col_i >= 0 && col_i < n_width) {
                    input_value = input[row_i * n_width + col_i];
                }
                float weight = d_M[(i + n) * m_width + (j + n)];
                p_value += input_value * weight;
            }
        }

        output[row_o * n_width + col_o] = p_value;
    }
}

__global__ void convolution_halo(float *input, float *output, int n_width, int m_width) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];

    int n = m_width / 2;

    int row_o = blockIdx.y * BLOCK_WIDTH + (threadIdx.y - n);
    int col_o = blockIdx.x * BLOCK_WIDTH + (threadIdx.x - n);

    int row_i = row_o;
    int col_i = col_o;

    if (row_i >= 0 && row_i < n_width && col_i >= 0 && col_i < n_width) {
        tile[threadIdx.y][threadIdx.x] = input[row_i * n_width + col_i];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (threadIdx.x >= n && threadIdx.x < TILE_WIDTH - n &&
    threadIdx.y >= n && threadIdx.y < TILE_WIDTH - n &&
        row_o < n_width && col_o < n_width) {

        float P = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                P += tile[threadIdx.y - n + i][threadIdx.x - n + j] *
                     d_M[i * MASK_WIDTH + j];
            }
        }

        output[row_o * n_width + col_o] = P;
    }
}

int main() {
    float h_N[N_WIDTH * N_WIDTH], h_O[N_WIDTH * N_WIDTH];
    float h_M[MASK_WIDTH * MASK_WIDTH] = {
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
    };

    const int n_size = N_WIDTH * N_WIDTH * sizeof(float);
    const int m_size = MASK_WIDTH * MASK_WIDTH * sizeof(float);

    for (int i = 0; i < N_WIDTH * N_WIDTH; i++) h_N[i] = 1.0f;

    float *d_N, *d_O;
    cudaMalloc(&d_N, n_size);
    cudaMalloc(&d_O, n_size);

    cudaMemcpy(d_N, h_N, n_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, h_M, m_size);

    dim3 dim_block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dim_grid((N_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (N_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < REPEAT; i++) {
        convolution<<<dim_grid, dim_block>>>(d_N, d_O, N_WIDTH, MASK_WIDTH);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_O, d_O, n_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_WIDTH * N_WIDTH; i++) {
        if (i % N_WIDTH == 0){
            printf("\n");
        }
        printf("%.0f ", h_O[i]);
    }

    printf("\n\ntotal time for %d runs %f ms\n", REPEAT, ms);
    printf("average cost: %f ms\n", ms/REPEAT);

    // ---------------------------- halo ------------------------------------
    printf("\n\ntesting halo...\n");

    float h_O_halo[N_WIDTH * N_WIDTH];
    float *d_O_halo;
    cudaMalloc(&d_O_halo, n_size);

    dim3 dim_block_halo(TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid_halo((N_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (N_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    cudaEvent_t start_halo, stop_halo;
    cudaEventCreate(&start_halo);
    cudaEventCreate(&stop_halo);

    cudaEventRecord(start_halo);
    for (int i = 0; i < REPEAT; i++) {
        convolution_halo<<<dim_grid_halo, dim_block_halo>>>(d_N, d_O_halo, N_WIDTH, MASK_WIDTH);
    }
    cudaEventRecord(stop_halo);
    cudaEventSynchronize(stop_halo);

    float ms_halo;
    cudaEventElapsedTime(&ms_halo, start_halo, stop_halo);

    cudaMemcpy(h_O_halo, d_O_halo, n_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_WIDTH * N_WIDTH; i++) {
        if (i % N_WIDTH == 0){
            printf("\n");
        }
        printf("%.0f ", h_O_halo[i]);
    }

    printf("\n\nhalo total time for %d runs %f ms\n", REPEAT, ms_halo);
    printf("halo average cost: %f ms\n", ms_halo/REPEAT);

    cudaFree(d_N);
    cudaFree(d_O);
    cudaFree(d_O_halo);
    return 0;
}