//
// Created by Jayden Hong on 2025-02-07.
//
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

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

    int n = 100;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if(!id) {
        printf("calculating number of primes from %d to %d ...\n", 0, n);
    }

    bool *local_mark = (bool *)malloc(n * sizeof(bool));
    bool *global_mark = (bool *) malloc(n * sizeof(bool));
    if(local_mark == NULL || global_mark == NULL) {
        printf("malloc memory error\n");
        MPI_Finalize();
        return 1;
    }
    local_mark[0] = local_mark[1] = global_mark[0] = global_mark[1] = true;
    for(int i = 2;i < n;i++) {
        local_mark[i] = false;
        global_mark[i] = false;
    }

    int limit_sqrt = (int)sqrt(n);
    for (int i = 2; i <= limit_sqrt; i++) {
        if (!local_mark[i]) {
            for (int multiple = i*i; multiple <= limit_sqrt; multiple += i) {
                local_mark[multiple] = true;
            }
        }
    }

    int index = 0;
    for (int i = 2; i <= limit_sqrt; i++) {
        if (!local_mark[i]) {
            if(index % p == id) {
                int first = limit_sqrt + i - limit_sqrt % i;
                for (int j = first; j < n; j += i) {
                    local_mark[j] = true;
                }
            }
            index++;
        }
    }

    MPI_Reduce(local_mark, global_mark, n, MPI_C_BOOL, MPI_LOR, 0, MPI_COMM_WORLD);
    int sum = 0;
    for(int i = 0; i < n; i++) {
        if(!global_mark[i]) {
            sum++;
        }
    }

    double end_time = MPI_Wtime();

    if (!id) {
        printf("find %d primes from 0 to %d\n", sum, n);
        printf("time cost: %10.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
