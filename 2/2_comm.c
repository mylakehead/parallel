#include <stdio.h>
#include <stdlib.h>
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

    int local_first_prime = -1;
    int local_last_prime  = -1;
    int local_max_gap     = 0;
    int prev_prime        = -1;

    for (int n = start_n; n < end_n; n++) {
        if (is_prime(n)) {
            if (prev_prime == -1) {
                local_first_prime = n;
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

    local_last_prime = prev_prime;

    int *rec_first_primes = NULL;
    int *rec_last_primes  = NULL;
    int *rec_max_gaps     = NULL;
    if (id == 0) {
        rec_first_primes = (int*)malloc(sizeof(int) * p);
        rec_last_primes  = (int*)malloc(sizeof(int) * p);
        rec_max_gaps     = (int*)malloc(sizeof(int) * p);
    }

    MPI_Gather(&local_first_prime, 1, MPI_INT, rec_first_primes,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_last_prime, 1, MPI_INT, rec_last_primes,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_max_gap, 1, MPI_INT, rec_max_gaps,  1, MPI_INT, 0, MPI_COMM_WORLD);

    if (id == 0) {
        if (rec_first_primes == NULL || rec_last_primes == NULL || rec_max_gaps == NULL) {
            printf("Memory allocation failed\n");
            MPI_Finalize();
            exit(1);
        }

        int global_max_gap = 0;
        for (int i = 0; i < p; i++) {
            if (rec_max_gaps[i] > global_max_gap) {
                global_max_gap = rec_max_gaps[i];
            }
        }

        for (int i = 0; i < p - 1; i++) {
            int last_p = rec_last_primes[i];
            int first_p_next = rec_first_primes[i + 1];
            if (last_p != -1 && first_p_next != -1) {
                int boundary_gap = first_p_next - last_p;
                if (boundary_gap > global_max_gap) {
                    global_max_gap = boundary_gap;
                }
            }
        }

        double end_time = MPI_Wtime();
        printf("parallel process largest gap: %d\n", global_max_gap);
        printf("parallel time: %f seconds\n", end_time - start_time);

        free(rec_first_primes);
        free(rec_last_primes);
        free(rec_max_gaps);
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
