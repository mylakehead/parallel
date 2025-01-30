//
// Created by Jayden Hong on 2025-01-29.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// NOLINTNEXTLINE(misc-no-recursion)
void quick_sort(int *arr, int left, int right) {
    if (left >= right)
        return;

    int pivot = arr[left];
    int i = left, j = right;

    while (i < j) {
        while (i < j && arr[j] >= pivot) j--;
        while (i < j && arr[i] <= pivot) i++;
        if (i < j) swap(&arr[i], &arr[j]);
    }
    swap(&arr[left], &arr[i]);

    quick_sort(arr, left, i - 1);
    quick_sort(arr, i + 1, right);
}

// NOLINTNEXTLINE(misc-no-recursion)
void parallel_quick_sort(int *arr, int n, int id, int p, MPI_Comm comm) {
    if (p == 1) {
        quick_sort(arr, 0, n - 1);
        return;
    }

    int pivot;
    if (id == 0) {
        pivot = arr[0];  // 选择第一个元素作为 pivot
    }

    // 广播 pivot 给所有进程
    MPI_Bcast(&pivot, 1, MPI_INT, 0, comm);

    // 分区
    int left_count = 0, right_count = 0;
    int *left = (int *)malloc(n * sizeof(int));
    int *right = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        if (arr[i] <= pivot) left[left_count++] = arr[i];
        else right[right_count++] = arr[i];
    }

    // 计算新的进程数
    int new_size = p / 2;
    MPI_Comm new_comm;
    int color = (id < new_size) ? 0 : 1;
    int key = id - (color == 1 ? new_size : 0);
    MPI_Comm_split(comm, color, key, &new_comm);

    if (color == 0) {
        parallel_quick_sort(left, left_count, id, new_size, new_comm);
    } else {
        parallel_quick_sort(right, right_count, id - new_size, p - new_size, new_comm);
    }

    // 进程合并数据
    int *sorted_data = (int *)malloc(n * sizeof(int));
    MPI_Gather(left, left_count, MPI_INT, sorted_data, left_count, MPI_INT, 0, comm);
    MPI_Gather(right, right_count, MPI_INT, sorted_data + left_count, right_count, MPI_INT, 0, comm);

    if (id == 0) {
        printf("Sorted data: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", sorted_data[i]);
        }
        printf("\n");
    }

    free(left);
    free(right);
    free(sorted_data);
    MPI_Comm_free(&new_comm);
}

int main(int argc, char *argv[]) {
    int id, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    printf("total processes: %d, process: %d is running\n", p, id);

    int n = 20;
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        printf ("malloc failed\n");
        MPI_Finalize();
        exit (1);
    }

    if (id == 0) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        srand(ts.tv_nsec ^ ts.tv_sec);

        for (int i = 0; i < n; i++) {
            arr[i] = (int) (random() % n);
            printf("%d ", arr[i]);
        }
        printf("\n");

        printf("process %d, %d testing data generated\n", id, n);
    }

    double start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    parallel_quick_sort(arr, n, id, p, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if(id == 0) {
        printf("process %d, parallel quick sort time: %f seconds\n", id, end - start);
    }

    free(arr);
    MPI_Finalize();

    if (id == 0) {
        int *arr_serial = (int *)malloc(n * sizeof(int));
        if (arr_serial == NULL) {
            printf ("malloc failed\n");
            MPI_Finalize();
            exit (1);
        }
        for (int i = 0; i < n; i++) {
            arr_serial[i] = arr[i];
        }
        clock_t start_serial = clock();
        quick_sort(arr_serial, 0, n - 1);
        clock_t end_serial = clock();

        printf("process %d, serial quick sort time: %f seconds\n", id,
               (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

        free(arr_serial);
    }

    return 0;
}
