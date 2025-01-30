//
// Created by Jayden Hong on 2025-01-27.
//

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>


// Fisher-Yates shuffle
void shuffle_array(int *array, int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = (int)(arc4random() % (i + 1));

        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void print_edges(unsigned char n, unsigned char f, bool cube) {
    if(cube) {
        printf("unused hypercube edge: ");
    }

    for (int i = 3; i >= 0; i--) {
        int bit = (n >> i) & 1;
        printf("%d", bit);
    }

    printf(" -> ");

    for (int i = 3; i >= 0; i--) {
        int bit = (f >> i) & 1;
        printf("%d", bit);
    }

    printf("\n");
}

// NOLINTNEXTLINE(misc-no-recursion)
void flip(unsigned char root, int *unreached, int size, unsigned char array[100][2], int *rows) {
    shuffle_array(unreached, size);

    for(int i = 0; i < size; i++) {
        unsigned char f = root ^ (1 << unreached[i]);

        array[*rows][0] = root;
        array[*rows][1] = f;
        (*rows)++;

        int k = size-i-1;
        int rest[k];
        for(int j = i + 1; j < size; j++) {
            rest[j-i-1] = unreached[j];
        }

        if(k > 0) {
            flip(f, rest, k, array, rows);
        }
    }
}

// NOLINTNEXTLINE(misc-no-recursion)
void cube(unsigned char root, const int *unreached, int size, unsigned char array[100][2], int *rows) {
    for(int i = 0; i < size; i++) {
        unsigned char f = root ^ (1 << unreached[i]);

        bool find = false;
        for(int j = 0;j < (*rows);j++) {
            if(array[j][0] == root && array[j][1] == f) {
                find = true;
            }
        }
        if(!find) {
            array[*rows][0] = root;
            array[*rows][1] = f;
            (*rows)++;
        }

        int k = size - 1;
        int rest[k];

        int m = 0;
        for (int j = 0; j < size; j++) {
            if (j != i) {
                rest[m++] = unreached[j];
            }
        }

        if (k > 0) {
            cube(f, rest, k, array, rows);
        }
    }
}

int main() {
    // the dimensions of hypercube
    int k = 4;
    int dims[k];
    for (int i = 0; i < k; i++) {
        dims[i] = i;
    }
    // only work with the lower four bits of data
    // you can change the root from 0b00000000 t0 0b00001111
    unsigned char root = 0b00000110;

    // to build a binomial tree started from one specific vertex
    // we need to walk through all the vertices
    // each time we can only flip one bit
    unsigned char array[100][2];
    int rows = 0;

    flip(root, dims, k, array, &rows);

    printf("edges of binomial tree: %d\n", rows);
    for(int i=0;i<rows;i++) {
        print_edges(array[i][0], array[i][1], false);
    }
    printf("\n");

    // build hypercube
    int cube_dims[k];
    for (int i = 0; i < k; i++) {
        cube_dims[i] = i;
    }
    unsigned char cube_root = 0b00000000;
    unsigned char cube_array[100][2];
    int cube_rows = 0;

    cube(cube_root, cube_dims, k, cube_array, &cube_rows);

    // printf("edges of hypercube tree: %d\n", cube_rows);
    for(int i=0;i<cube_rows;i++) {
        //print_edges(cube_array[i][0], cube_array[i][1], false);
    }
    // printf("\n");

    // compare
    int total = 0;
    for(int i=0;i<cube_rows;i++) {
        bool find = false;
        for(int j=0;j<rows;j++) {
            if((cube_array[i][0] == array[j][0] && cube_array[i][1] == array[j][1]) ||
            (cube_array[i][0] == array[j][1] && cube_array[i][1] == array[j][0])) {
                find = true;
            }
        }
        if(!find) {
            print_edges(cube_array[i][0], cube_array[i][1], true);
            total++;
        }
    }
    printf("all %d unused\n", total);

    return 0;
}
