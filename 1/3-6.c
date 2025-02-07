//
// Created by Jayden Hong on 2025-01-28.
//

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, local_value, reduced_value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_value = rank + 1;

    printf("process %d local: %d\n", rank, local_value);

    MPI_Reduce(&local_value, &reduced_value, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("all sum: %d\n", reduced_value);
    }

    MPI_Finalize();
    return 0;
}

