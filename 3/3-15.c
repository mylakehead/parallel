//
// Created by Jayden Hong on 2025-01-29.
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>


typedef struct {
    double x, y;
} Point;

typedef struct {
    double distance;
    int index;
} Distance;

double distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

int main(int argc, char** argv) {
    int id, p;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    Point station = {10.0, 10.0};
    int n = 100000000;
    Point *houses = malloc(n * sizeof(Point));
    if (houses == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    if (id == 0) {
        for (int i = 0; i < n; i++) {
            houses[i].x = (double)arc4random() / UINT32_MAX * 100.0;
            houses[i].y = (double)arc4random() / UINT32_MAX * 100.0;
        }
    }

    double start_time, end_time;

    MPI_Barrier(MPI_COMM_WORLD);
    if(id == 0) {
        start_time = MPI_Wtime();
    }

    size_t local_size = n / p;
    Point *local_houses = malloc(local_size * sizeof(Point));
    if (local_houses == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    clock_t start_serial = clock();
    MPI_Scatter(houses, (int)(local_size * sizeof(Point)), MPI_BYTE, local_houses, (int)(local_size * sizeof(Point)), MPI_BYTE,
                0, MPI_COMM_WORLD);
    clock_t end_serial = clock();
    printf("process %d, scatter time: %f seconds\n", id, (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

    Distance local = {DBL_MAX, -1};

    start_serial = clock();
    for (int i = 0; i < local_size; i++) {
        double dist = distance(local_houses[i], station);
        if (dist < local.distance) {
            local.distance = dist;
            local.index = id * (int)local_size + i;
        }
    }
    end_serial = clock();
    printf("process %d, serial time: %f seconds\n", id, (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

    start_serial = clock();
    Distance global;
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    end_serial = clock();
    printf("process %d, reduce time: %f seconds\n", id, (double)(end_serial - start_serial) / CLOCKS_PER_SEC);

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        end_time = MPI_Wtime();
        printf("the closest house is at index %d with x: %f, y: %f, distance %lf, execution time: %f seconds\n", global.index,
               houses[global.index].x, houses[global.index].y, global.distance, end_time-start_time);
    }

    free(local_houses);
    free(houses);

    MPI_Finalize();
    return 0;
}
