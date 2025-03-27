#include "mpi.h"
#include <stdio.h>

int main (int argc, char *argv[])
{
    int num_tasks, rank, dest, tag, source;
    char in_msg, out_msg = 'x';

    MPI_Status Stat;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (num_tasks > 2) {
            printf("number of tasks = %d. only 2 needed. Ignoring extra...\n", num_tasks);
        }

        dest = rank + 1;
        source = dest;
        tag = rank;

        // 小消息（通常 < 64KB）：
        // 标准模式（eager protocol）：MPI 可能会在内部缓冲消息，MPI_Send 立即返回。
        // 大消息（通常 > 64KB）：
        // 同步模式（rendezvous protocol）：MPI 可能会等待 MPI_Recv 开始接收，然后 MPI_Send 才会返回，可能阻塞。
        printf("rank %d try sending message to rank %d with tag %d\n", rank, dest, tag);
        MPI_Send(&out_msg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);

        printf("rank %d try receiving message from rank %d with tag %d\n", rank, source, tag);
        MPI_Recv(&in_msg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);
        printf("Received from task %d...\n",source);
    }
    else if (rank == 1) {
        dest = rank - 1;
        source = dest;
        // MPI_Recv with MPI_ANY_TAG
        // or tag = rank -1
        // can solve this problem
        tag = rank;

        printf("rank %d try receiving message from rank %d with tag %d\n", rank, source, tag);
        MPI_Recv(&in_msg, 1, MPI_CHAR, source, tag, MPI_COMM_WORLD, &Stat);

        printf("rank %d try sending message to rank %d with tag %d\n", rank, dest, tag);
        MPI_Send(&out_msg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}