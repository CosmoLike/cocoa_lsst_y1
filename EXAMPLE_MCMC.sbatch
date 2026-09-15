#!/bin/bash
#SBATCH --job-name=MC
#SBATCH --output=MC-%j.out
#SBATCH -e MC-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=YYY
#SBATCH --account=XXX

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
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; 
export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

#ulimit -u 2000000 # require line when nmpi is high
"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  --mca pml ob1 --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --report-bindings \
  --rank-by slot --map-by slot:pe=${OMP_NUM_THREADS} \
  cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -r
