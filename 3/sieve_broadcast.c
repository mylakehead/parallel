//
// Created by Jayden Hong on 2025-02-07.
//
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_LOW(id, p, n)  ( ((id) * (n)) / (p) )
#define BLOCK_HIGH(id, p, n) ( BLOCK_LOW((id)+1, p, n) - 1 )
#define BLOCK_SIZE(id, p, n) ( BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1 )
#define BLOCK_OWNER(index,p,n) (((p)*(index)+1)-1)/(n))


int main(int argc, char **argv) {
    int id, p;

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = 100000;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if(!id) {
        printf("calculating number of primes from %d to %d ...\n", 0, n);
    }

    int low_value = 2 + BLOCK_LOW(id, p, n - 1);
    int high_value = 2 + BLOCK_HIGH(id, p, n - 1);
    int size = BLOCK_SIZE(id, p, n - 1);

    int proc0_size = (n - 1) / p;
    if ((2 + proc0_size) < (int) sqrt((double) n)) {
        if (!id) {
            printf("too many processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    char  *marked;
    marked = (char *)malloc(size);
    if (marked == NULL) {
        printf("process %d can not allocate memory\n", id);
        MPI_Finalize();
        return 1;
    }
    for (int i = 0; i < size; i++) {
        marked[i] = 0;
    }

    int index;
    if (!id) {
        index = 0;
    }

    int i, first;
    int count, global_count;
    int prime = 2;
    do {
        if (prime * prime > low_value) {
            first = prime * prime - low_value;
        } else {
            if ((low_value % prime) == 0) {
                first = 0;
            } else {
                first = prime - (low_value % prime);
            }
        }

        for (i = first; i < size; i += prime) {
            marked[i] = 1;
        }

        if (!id) {
            while (marked[++index]) {
            }
            prime = index + 2;
        }

        MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= n);

    count = 0;
    for (i = 0; i < size; i++) {
        if (!marked[i]) {
            count++;
        }
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (!id) {
        printf("find %d primes from 0 to %d\n", global_count, n);
        printf("time cost: %10.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
