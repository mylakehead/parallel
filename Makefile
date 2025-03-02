default: mpicc

mpicc:
	mpicc -std=c99 ?.c -o ?
	mpicc ./3/3-9.c -o 3-9 -I/usr/local/include -L/usr/local/lib
	/opt/mpich2/gnu/bin/mpicc -std=c99 -o 2_comm ./2_comm.c -lm

a2:
	mpicc ./2/2_comm.c -o ./2/2_comm_exe -I/usr/local/include -L/usr/local/lib
	mpicc ./2/2_overlapping.c -o ./2/2_overlapping_exe -I/usr/local/include -L/usr/local/lib

a2_wesley:
	/opt/mpich2/gnu/bin/mpicc -std=c99 -o ./2/2_comm_exe ./2/2_comm.c -lm
	/opt/mpich2/gnu/bin/mpicc -std=c99 -o ./2/2_overlapping_exe ./2/2_overlapping.c -lm

a3_wesley:
	/opt/mpich2/gnu/bin/mpicc -std=c99 -o ./3/sieve_reduce_exe ./3/sieve_reduce.c -lm

a4_wesley:
	/opt/mpich2/gnu/bin/mpicc -std=c99 -o ./4/latency_exe ./4/latency.c -lm

run:
	qsub -l nodes=2:ppn=1 ./sieve_reduce.sh


clean:
	rm -rf *.dSYM
	rm -f ./*/*_exe