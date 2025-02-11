EXE = 2_comm 2_overlapping 3-15 3-5

default: mpicc

mpicc:
	mpicc -std=c99 ?.c -o ?
	mpicc ./3/3-9.c -o 3-9 -I/usr/local/include -L/usr/local/lib

run:
	mpiexec -n 6 ./2_comm
	mpiexec -n 6 ./sieve_broadcast 100000000

clean:
	rm -rf *.dSYM
	rm -f $(EXE)