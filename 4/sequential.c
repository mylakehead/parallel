//
// Created by Jayden Hong on 2025-02-05.
//

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define LIMIT 1000000


int main() {
    clock_t start_serial = clock();

    // 1) Allocate space for the sieve
    bool *is_prime = malloc(sizeof(bool) * LIMIT);
    for (int i = 2; i < LIMIT; i++) {
        is_prime[i] = true;
    }
    is_prime[0] = false;
    is_prime[1] = false;

    // 2) Sieve of Eratosthenes
    int limit_sqrt = (int) sqrt(LIMIT);
    for (int p = 2; p <= limit_sqrt; p++) {
        if (is_prime[p]) {
            for (int multiple = p*p; multiple < LIMIT; multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }

    // 3) Collect primes into a dynamic array
    int *primes = malloc(sizeof(int) * LIMIT); // Over-allocating for simplicity
    int count = 0;
    for (int i = 2; i < LIMIT; i++) {
        if (is_prime[i]) {
            primes[count++] = i;
        }
    }

    // 4) Find largest gap
    int largest_gap = 0;
    for (int i = 0; i < count - 1; i++) {
        int gap = primes[i+1] - primes[i];
        if (gap > largest_gap) {
            largest_gap = gap;
        }
    }

    printf("Largest gap between consecutive primes under %d is %d.\n", LIMIT, largest_gap);

    // Clean up
    free(is_prime);
    free(primes);

    clock_t end_serial = clock();
    printf("process time: %f seconds\n", (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

    return 0;
}
