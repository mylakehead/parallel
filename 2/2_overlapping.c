#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#define LIMIT 100000000


bool is_prime(int n) {
    if (n < 2) return false;
    int limit = (int) sqrt(n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int p, id;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double start_time = MPI_Wtime();

    int chunk_size = LIMIT / p;
    int start_n = id * chunk_size;
    int end_n   = (id == p - 1) ? LIMIT : (start_n + chunk_size);

    if (start_n < 2) {
        start_n = 2;
    }

    int local_max_gap     = 0;
    int prev_prime        = -1;

    for (int n = start_n; n < end_n; n++) {
        if (is_prime(n)) {
            if (prev_prime == -1) {
                prev_prime        = n;
            } else {
                int gap = n - prev_prime;
                if (gap > local_max_gap) {
                    local_max_gap = gap;
                }
                prev_prime = n;
            }
        }
    }

    int next = end_n;
    while (next < LIMIT){
        if(is_prime(next)){
            if (prev_prime != -1) {
                int gap = next - prev_prime;
                if (gap > local_max_gap) {
                    local_max_gap = gap;
                }
            }
            break;
        }
        next++;
    }

    int global_max_gap = 0;
    MPI_Reduce(&local_max_gap, &global_max_gap, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (id == 0) {
        printf("largest gap = %d\n", global_max_gap);
        printf("MPI overlapping time = %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();

    if(id == 0) {
        clock_t start_serial = clock();

        int largest_gap = 0;
        prev_prime = 2;

        for (int n = 2; n < LIMIT; n++) {
            if (is_prime(n)) {
                int gap = n - prev_prime;
                if (gap > largest_gap) {
                    largest_gap = gap;
                }
                prev_prime = n;
            }
        }

        clock_t end_serial = clock();

        printf("sequential process largest gap: %d\n", largest_gap);
        printf("sequential process time: %f seconds\n", (double)(end_serial - start_serial) / CLOCKS_PER_SEC);
    }

    return 0;
}
