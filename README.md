# parallel

### run in local environment(macOS)
```shell
make a2
mpiexec -n 6 ./2/2_comm_exe
```

### run in wesley environment
```shell
make a2_wesley
cd 2
qsub -l nodes=1:ppn=1 ./2_comm.sh
```
