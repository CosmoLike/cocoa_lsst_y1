#!/bin/bash

#SBATCH --job-name=PR
#SBATCH --output=PR-%A_%a.out
#SBATCH -e PR-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --ntasks-per-node=94
#SBATCH --ntasks-per-socket=47
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
declare -i NY=100 # so from 0-99 is all about PROFILE1, 100-199 all about PROFILE2
if [ -z "${ROOTDIR}" ]; then
  declare -i tmp=$(( $SLURM_ARRAY_TASK_ID % NY ))
  sleep $(( 5 + $tmp*15 )) # help avoid different scripts running start_cocoa.sh simultaneously
                           # clash can still happen when one script runs start_cocoa.sh, the other
                           # script is reading a file (start_cocoa.sh invokes stop_cocoa.sh)
                           # which delete links and ROOTDIR so the running script can find file/likelihood
  source ~/.bashrc
  conda activate cocoa
  source start_cocoa.sh
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 
export MKL_NUM_THREADS=1
ulimit -u 2000000 # require line when nmpi is high

# the parameters are 
# As_1e9    ns        H0      omegab  omegam  
# LSST_DZ_S1 LSST_DZ_S2 LSST_DZ_S3 LSST_DZ_S4 LSST_DZ_S5 
# LSST_A1_1 LSST_A1_2 
# LSST_M1 LSST_M2 LSST_M3 LSST_M4 LSST_M5
# We advise factor ~ 3 for parameters that are well constrained by the data (...) 
# If a parameter is poorly constrained or cov is not given, we recommend factor ~< 1
declare -a factor=( "3" "1.0" "1.0" "1.0" "3" 
                    "3" "3" "3" "3" "3" 
                    "3" "3" "3" "3" "3" 
                    "3" "3" )              
declare -a factor2=( "3" "3.0" "3.0" "3.0" "3" 
                     "3" "3" "3" "3" "3" 
                     "3" "3" "3" "3" "3" 
                     "3" "3" )
declare -a grid=( factor1 factor2 )
declare -a nstws=( "400" "700" )
declare -i i=$(( $SLURM_ARRAY_TASK_ID / NY + 1 ))
declare -i j=$(( $SLURM_ARRAY_TASK_ID % NY ))
(( i < ${#grid[@]} )) || exit 0 # padding: skip unused slots
declare -n row="${grid[$i]}"
declare nstw="${nstws[$i]}"
(( j < ${#row[@]} )) || exit 0  # padding: skip unused slots
declare factor="${row[$j]}"

"${CONDA_PREFIX}"/bin/mpirun -n ${SLURM_NTASKS} --oversubscribe \
  -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR -x OMP_NUM_THREADS \
  -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC -x CLIK_PLUGIN \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
  --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
  --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
  python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE$((i+1)).py \
    --root ./projects/lsst_y1/ \
    --cov "chains/EXAMPLE_EMUL_MCMC$((i+1)).covmat" \
    --outroot "EXAMPLE_EMUL_PROFILE$i" \
    --factor "$factor" \
    --nstw $nstw \
    --numpts 15 \
    --profile "${j}" \
    --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN$((i+1)).txt"
