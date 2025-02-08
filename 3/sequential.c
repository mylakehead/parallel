//
// Created by Jayden Hong on 2025-02-07.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>


int main() {
    int n;
    printf("Enter the upper limit (n): ");
    scanf("%d", &n);

    bool *is_prime = (bool *)malloc((n + 1) * sizeof(bool));
    for (int i = 0; i <= n; i++) {
        is_prime[i] = true;
    }
    is_prime[0] = is_prime[1] = false;

    clock_t start = clock();

    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }

    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int total = 0;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            total++;
        }
    }
    printf("primes %d:\n", total);
    printf("time cost: %.3f ms\n", elapsed_time);

    free(is_prime);

    return 0;
}
