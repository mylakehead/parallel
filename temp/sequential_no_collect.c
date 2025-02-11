//
// Created by Jayden Hong on 2025-02-05.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

int main() {
    clock_t start_serial = clock();

    const int LIMIT = 1000000;
    bool *is_prime = malloc(LIMIT * sizeof(bool));

    // 1) 初始化数组，默认全部标记为 true（认为是质数），
    //    然后再把 0 和 1 标记为 false。
    for (int i = 0; i < LIMIT; i++) {
        is_prime[i] = true;
    }
    is_prime[0] = false;
    is_prime[1] = false;

    // 2) 使用筛法标记合数
    int limit_sqrt = (int) sqrt(LIMIT);
    for (int p = 2; p <= limit_sqrt; p++) {
        if (is_prime[p]) {
            // 将 p 的倍数标记为 false
            for (int multiple = p * p; multiple < LIMIT; multiple += p) {
                is_prime[multiple] = false;
            }
        }
    }

    // 3) 遍历 is_prime[]，直接计算相邻质数之间的 gap
    int largest_gap = 0;
    int prev_prime = -1;  // 用于存储上一个质数

    for (int i = 2; i < LIMIT; i++) {
        if (is_prime[i]) {
            if (prev_prime != -1) {
                int gap = i - prev_prime;
                if (gap > largest_gap) {
                    largest_gap = gap;
                }
            }
            prev_prime = i;
        }
    }

    printf("Largest gap between consecutive primes < %d is %d.\n", LIMIT, largest_gap);

    // 释放内存
    free(is_prime);

    clock_t end_serial = clock();
    printf("process time: %f seconds\n", (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

    return 0;
}

