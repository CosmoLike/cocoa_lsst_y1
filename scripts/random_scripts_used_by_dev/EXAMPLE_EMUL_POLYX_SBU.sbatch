#!/bin/bash

#SBATCH --job-name=POLY
#SBATCH --output=POLY-%A_%a.out
#SBATCH -e POLY-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=85
#SBATCH --cpus-per-task=1
#SBATCH --partition=hbm-long-96core
#SBATCH --time=36:00:00

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
if [ -z "${ROOTDIR}" ]; then
  sleep $(( 5 + SLURM_ARRAY_TASK_ID*25 )) # help avoid different scripts running start_cocoa.sh simultaneously
                                          # clash can still happen when one script runs start_cocoa.sh, the other
                                          # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                                          # which delete links and ROOTDIR so the running script can find file/likelihood
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 
export MKL_NUM_THREADS=1

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR -x OMP_NUM_THREADS \
  -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC -x CLIK_PLUGIN \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
  --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
  --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
  cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY${SLURM_ARRAY_TASK_ID}.yaml -r
