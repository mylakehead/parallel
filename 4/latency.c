//
// Created by Jayden Hong on 2025-02-09.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if(argc < 3) {
        printf("please input sending times and message length(bytes)\n");
        return 1;
    }
    int times = atoi(argv[1]);
    int length = atoi(argv[2]);
    printf("%d parameters input, sending times: %d, length: %d (bytes)\n", argc, times, length);

    MPI_Init(&argc, &argv);
    int id, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p != 2) {
        if (id == 0) {
            printf("2 processes required.\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *buffer = (char *) malloc(length);
    if (!buffer) {
        printf("malloc memory failed\n");
        MPI_Finalize();
        return 1;
    }

    double total_time = 0.0;
    for (int i = 0; i < times; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        double t_start = MPI_Wtime();
        if (id == 0) {
            MPI_Send(buffer, length, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, length, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double t_end = MPI_Wtime();
            total_time += (t_end - t_start);
        }
        else {
            MPI_Recv(buffer, length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (id == 0) {
        double avg_rtt = total_time / times;
        double one_way = avg_rtt / 2.0;

        printf("Message p (bytes) = %d,  Avg RTT (s) = %g,  One-way time (s) = %g\n", length, avg_rtt, one_way);
    }

    free(buffer);

    MPI_Finalize();
    return 0;
}
