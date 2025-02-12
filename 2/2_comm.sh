cd $PBS_O_WORKDIR
nprocs=`wc -l < $PBS_NODEFILE`
echo nprocs = $nprocs
echo
echo Starting job
/opt/mpich2/gnu/bin/mpiexec.hydra -np $nprocs -f $PBS_NODEFILE ./2_comm_exe
echo Job complete