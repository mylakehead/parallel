#include <stdio.h>

int main() {
    int arr[3][2][4][4] = {
            {{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24},{25,26,27,28},{29,30,31,32}}},
            {{{41,42,43,44},{45,46,47,48},{49,410,411,412},{413,414,415,416}},{{417,418,419,420},{421,422,423,424},{425,426,427,428},{429,430,431,432}}},
            {{{51,52,53,54},{55,56,57,58},{59,510,511,512},{513,514,515,516}},{{517,518,519,520},{521,522,523,524},{525,526,527,528},{529,530,531,532}}}
    };

            // 通过指针遍历
            int *ptr = (int *)arr;  // 将多维数组转换为一维指针
            int size = 2 * 3 * 4;   // 计算总元素个数

            for (int i = 0; i < size; i++) {
                printf("%d ", *(ptr + i));
            }

            printf("\n");
            return 0;
    }

    /*
     %%writefile hello.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // 每个 Block 计算 16×16 的 Tile

__global__ void queryKeyMultiplication(float* Q, float* K, float* A, int seq_len, int d_model) {
    // 定义共享内存 Tile
    __shared__ float Q_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float K_shared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; // 当前线程计算的行索引
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; // 当前线程计算的列索引

    float sum = 0.0;

    // 遍历 d_model 维度，分块加载
    for (int t = 0; t < (d_model + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // 加载 Q 和 K 的 Tile 到共享内存
        if (row < seq_len && t * BLOCK_SIZE + threadIdx.x < d_model)
            Q_shared[threadIdx.y][threadIdx.x] = Q[row * d_model + (t * BLOCK_SIZE + threadIdx.x)];
        else
            Q_shared[threadIdx.y][threadIdx.x] = 0.0;

        if (col < seq_len && t * BLOCK_SIZE + threadIdx.y < d_model)
            K_shared[threadIdx.y][threadIdx.x] = K[col * d_model + (t * BLOCK_SIZE + threadIdx.y)];
        else
            K_shared[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();  // 确保所有线程加载完数据

        // 计算 Tile 内的部分矩阵乘法
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += Q_shared[threadIdx.y][i] * K_shared[i][threadIdx.x];
        }
        __syncthreads();  // 确保所有线程计算完 Tile
    }

    // 存储最终计算结果
    if (row < seq_len && col < seq_len) {
        A[row * seq_len + col] = sum;
    }
}

int main() {
    int seq_len = 64, d_model = 128;
    size_t size_QK = seq_len * d_model * sizeof(float);
    size_t size_A = seq_len * seq_len * sizeof(float);

    float *h_Q = (float*)malloc(size_QK);
    float *h_K = (float*)malloc(size_QK);
    float *h_A = (float*)malloc(size_A);

    // 初始化数据
    for (int i = 0; i < seq_len * d_model; i++) {
        h_Q[i] = 0.01f * (i % 10);
        h_K[i] = 0.02f * (i % 10);
    }

    float *d_Q, *d_K, *d_A;
    cudaMalloc(&d_Q, size_QK);
    cudaMalloc(&d_K, size_QK);
    cudaMalloc(&d_A, size_A);

    cudaMemcpy(d_Q, h_Q, size_QK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_QK, cudaMemcpyHostToDevice);

    // 计算 Grid & Block 配置
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动 CUDA Kernel
    queryKeyMultiplication<<<gridDim, blockDim>>>(d_Q, d_K, d_A, seq_len, d_model);

    cudaMemcpy(h_A, d_A, size_A, cudaMemcpyDeviceToHost);

    // 输出部分结果
    printf("A[0][0] = %f\n", h_A[0]);
    printf("A[1][1] = %f\n", h_A[1 * seq_len + 1]);

    // 释放内存
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_A);
    free(h_Q);
    free(h_K);
    free(h_A);

    return 0;
}
     */