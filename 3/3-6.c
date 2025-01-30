//
// Created by Jayden Hong on 2025-01-28.
//

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, local_value, reduced_value;

    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取当前进程号
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取进程总数

    // 每个进程的本地数据
    local_value = rank + 1; // 假设进程 i 的数据为 i+1

    printf("进程 %d 的本地值: %d\n", rank, local_value);

    // 使用 MPI_Reduce 进行归约
    MPI_Reduce(&local_value, &reduced_value, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 只有 root 进程（rank 0）会接收到归约的最终结果
    if (rank == 0) {
        printf("所有进程的归约和: %d\n", reduced_value);
    }

    // 关闭 MPI
    MPI_Finalize();
    return 0;
}

