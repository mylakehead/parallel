EXE = 2_comm 2_overlapping

default: mpicc

mpicc:
	mpicc ./3/3-9.c -o 3-9 -I/usr/local/include -L/usr/local/lib

run:
	mpiexec -n 6 ./2_comm

clean:
	rm -rf *.dSYM
	rm -f $(EXE)