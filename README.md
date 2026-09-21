# Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment out the following lines in `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
>     (...)
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export LSST_Y1_URL="https://github.com/CosmoLike/cocoa_lsst_y1.git"
>     export LSST_Y1_NAME="lsst_y1"
>     #Pin the project version with at most one of the keys below (COMMIT, BRANCH, or TAG).
>     #If more than one is set, COMMIT wins over BRANCH, and BRANCH wins over TAG.
>     #If none is set, Cocoa loads the latest commit on the repository default branch.
>     #export LSST_Y1_GIT_BRANCH="dev"
>     export LSST_Y1_GIT_COMMIT="1abe548281296196dabee7b19e31c56f324eda38"
>     #export LSST_Y1_GIT_TAG="v4.0-beta17"

> [!NOTE]
> In case users need to rerun `setup_cocoa.sh`, Cocoa will not download previously installed packages, cosmolike projects, or large datasets, unless the following keys are set on `set_installation_options.sh`
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # ------------------------------------------------------------------------------
>     # OVERWRITE_EXISTING_XXX_CODE=1 -> setup_cocoa overwrites existing PACKAGES ----
>     # overwrite: delete the existing PACKAGE folder and install it again -----------
>     # redownload: delete the compressed file and download data again ---------------
>     # These keys are only relevant if you run setup_cocoa multiple times -----------
>     # ------------------------------------------------------------------------------
>     (...)
>     export OVERWRITE_EXISTING_ALL_PACKAGES=1    # except cosmolike projects
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 # dangerous (possible loss of uncommitted work)
>                                                 # if unset, users must manually delete cosmolike projects
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1      # warning: some data is many GB
>

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile the lsst-y1 project)
>
>       source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
> 
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project lsst-y1 contains jupyter notebook examples located at `projects/lsst_y1`.

To run the example

 **Step :one:**: activate the Cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/lsst_y1` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning] 
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)
     
        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f


# Baryonic feedback on EXAMPLE_EVALUATE1 <a name="lsst_y1_baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback suppression to the
matter power spectrum via the `bfmt` theory block (SP(k), BCEmu, Flamingo, BACCOemu,
or BCemu2025). By default, the example runs without feedback.

**Step :one:**: ensure the lines below are commented out in `set_installation_options.sh`
before running `setup_cocoa.sh` and `compile_cocoa.sh`. *By default, these lines should
be commented out, but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt` theory block and select
the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the `lsst_y1.cosmic_shear`
likelihood block.

**Step :four:**: uncomment the selected model's parameters in the `params` block and in
the `sampler: evaluate: override` block (the example carries a commented block for each
model).

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and the `bfmt`
> options, see `Cocoa/external_modules/code/baryon_suppression/README.md`.

# Running ML emulators <a name="cobaya_base_code_examples_emul"></a>

Cocoa contains a few transformer- and CNN-based neural network emulators capable of simulating the CMB, cosmolike outputs, matter power spectrum, and distances. We provide a few scripts that exemplify their API. To run them, users ensure the following lines are commented out in `set_installation_options.sh` before running the `setup_cocoa.sh` and `compile_cocoa.sh`. By default, these lines should be commented out, but it is worth checking.

      [Adapted from Cocoa/set_installation_options.sh shell script] 
      # insert the # symbol (i.e., unset these environmental keys  on `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              #SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1            
      #export IGNORE_LIPOP_LIKELIHOOD_CODE=1     # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_LIPOP_CMB_DATA=1           
      #export IGNORE_ACTDR6_CODE=1               # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_ACTDR6_DATA=1         
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_NAUTILUS1.py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_POLY1.yaml
      #export IGNORE_GETDIST_CODE=1              # to run EXAMPLE_TENSION_METRICS.ipynb
      #export IGNORE_TENSIOMETER_CODE=1          # to run EXAMPLE_TENSION_METRICS.ipynb

> [!TIP]
> What if users have not configured ML-related keys before sourcing `setup_cocoa.sh`?
> 
> Answer: Comment the keys below before rerunning `setup_cocoa.sh`.
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # These keys are only relevant if you run setup_cocoa multiple times
>     #export OVERWRITE_EXISTING_ALL_PACKAGES=1    
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Ensure OpenMP is **OFF**.

    export OMP_NUM_THREADS=1

 **Step :three:** Run `cobaya-run` on the first emulator example following the commands below.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_EVALUATE1.yaml -f

  - macOS (arm)
 
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC1.yaml -r

> [!Note]
> Below is the cobaya timing of the average data vector computation time
> in a Metropolis-Hastings chain (`EXAMPLE_EMUL_MCMC1`, no slow/fast decomposition) on a macOS M2 Pro. Our emulators do take advantage of the CPU-GPU integration on Apple MX chips.
>
> Cobaya output: `lsst_y1.cosmic_shear : 0.00423518 s (200828 evaluations, 850.543 s total)`
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC2.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC2.yaml -r
      
> [!Note]
> The examples below may require a large number of MPI workers. Before running them, it may be necessary to increase 
> the limit of threads that can be created (at UofA HPC type `ulimit -u 1000000`), otherwise users 
> may encounter the error `libgomp: Thread creation failed`

- **PolyChord**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY1.yaml -r

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY1.yaml -r
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY2.yaml -r

  - macOS (arm)
 
        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY2.yaml -r

> [!NOTE]
> **Running on more than one node.** The flag `--mca btl vader,tcp,self` works unchanged across
> nodes: Open MPI picks the transport per pair of ranks, using shared memory (`vader`) within a
> node and TCP between nodes. Three things deserve attention on multi-node runs:
>
> 1. **Network interface.** The TCP layer must not select an interface that is not routable
>    between compute nodes. The flag `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0` excludes
>    the common offenders. TCP bandwidth is not a limitation for our workloads, which exchange
>    small, infrequent MPI messages.
>
> 2. **Environment.** Ranks on remote nodes must see Cocoa's environment (`ROOTDIR`, `PATH`,
>    `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, the OpenMP/BLAS thread settings, and
>    `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN`). Slurm forwards the submitting environment
>    automatically; the explicit `-x` flags in our sbatch templates repeat this so the
>    scripts also work under ssh-based launchers. No other Cocoa installation flags are read at runtime.
>
> 3. **Slurm geometry.** Keep `ntasks-per-node` × `cpus-per-task` no larger than the cores per
>    node, and use `--map-by numa:pe=${OMP_NUM_THREADS}` so each rank reserves the cores its
>    OpenMP threads will use.

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

The `Nautilus`, `Minimizer`, `Profile`, and `Emcee` scripts below contain an internally defined `yaml_string` that specifies priors, 
likelihoods, and the theory code, all following Cobaya Conventions.

- **Nautilus**:

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
            --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS1" \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  The Colab example [Test Nautilus](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1)_Test_Nautilus.ipynb) illustrates how stable Nautilus results are as a function of `nlive` 

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS2"  \
            --maxfeval 850000 --nlive 3072 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS2"  \
            --maxfeval 850000 --nlive 3072 --neff 15000 \
            --flive 0.01 --nnetworks 5
    
  What if the user runs a `Nautilus` chain with `maxeval` insufficient for producing `neff` samples? `Nautilus` saves the chain checkpoint at `chains/outroot_checkpoint.hdf5`.

- **Emcee**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE1" \
            --maxfeval 1000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE1" \
            --maxfeval 1000000
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE2" \
            --maxfeval 2000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE2" \
            --maxfeval 2000000
    
  The number of steps per MPI worker is $n_{\rm sw} =  {\rm maxfeval}/n_{\rm w}$,
  with the number of walkers being $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.
  For proper convergence, each walker should traverse 50 times the autocorrelation length ($\tau$),
  which is provided in the header of the output chain file. A reasonable rule of thumb is to assume
  $\tau > 200$ and therefore set ${\rm maxfeval} > 10,000 \times n_{\rm w}$.
  Finally, our code sets burn-in (per walker) at $5 \times \tau$.

  With these numbers, users may ask when `Emcee` is preferable to `Metropolis-Hastings`?
  Here are a few numbers based on our `Planck CMB (l < 396) + SN + BAO + LSST-Y1` test case.
  1) `MH` achieves convergence with $n_{\rm sw} \sim 150,000$ (number of steps per walker), but only requires four walkers.
  2) `Emcee` has $\tau \sim 300$, so it requires $n_{\rm sw} \sim 15,000$ when running with $n_{\rm w}=114$.
  
  Conclusion: `Emcee` requires $\sim 3$ more evaluations in this case, but the number of evaluations per MPI worker (assuming one MPI worker per walker) is reduced by $\sim 10$.
  Therefore, `Emcee` seems well-suited for chains where the evaluation of a single cosmology is time-consuming (and there is no slow/fast decomposition).

  What if the user runs an `Emcee` chain with `maxeval` insufficient for convergence? `Emcee` saves the chain checkpoint at `chains/outroot.h5`.

- **Sampler Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[2].py`.   The Google Colab notebooks [Example Sampler Comparison (LSST-Y1 only)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1).ipynb) and
  [Example Sampler Comparison (LSST+Others)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1)_Sampler_Comparison_2.ipynb) can also reconstruct a similar version of these figures.

  <p align="center">
  <img width="750" height="750" alt="project_lsst_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/ffc72bb0-1843-4a55-9a69-ca4c7d6b34c2" />
  </p>

  Another Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - $n_{\rm param} = 38$:

  <p align="center">
  <img width="750" height="750" alt="project_lsst_plot_sampler_comparison_2" src="https://github.com/user-attachments/assets/5bd7318e-864e-439d-9c3c-eaf07e267654" />
  </p>

- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225) 

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN1" \
            --nstw 350

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN1" \
            --nstw 350
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN2" \
            --nstw 750

  - macOS (arm)

        mpirun -n 12 \
          --oversubscribe python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE2.py \
          --root ./projects/lsst_y1/ \
          --outroot "EXAMPLE_EMUL_MIN2" \
          --nstw 750
    
  The number of steps per Emcee walker per temperature is $n_{\rm stw}$,
  and the number of walkers is $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.
  The minimum number of total evaluations is $3n_{\rm params} \times n_{\rm T} \times n_{\rm stw}$, which can be distributed among $n_{\rm MPI} = 3n_{\rm params}$ MPI processes for faster results.

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_MIN_COMPARE_CONV[2].py`

  <p align="center">
  <img width="750" height="750" alt="Screenshot 2025-08-12 at 8 36 33 PM" src="https://github.com/user-attachments/assets/31c36592-2d6c-4232-b5b4-5f686f9f2b8e" />
  </p>

  In our testing, $n_{\rm stw} \sim 200$ worked reasonably well up to $n_{\rm param} \sim \mathcal{O}(10)$.
  Below we show a case with $n_{\rm param} = 38$ that illustrates the need for performing convergence tests on a case-by-case basis.
  In this example, the total number of evaluations for a reliable minimum is approximately $319,200$ ($n_{\rm stw} \sim 700$), distributed among $n_{\rm MPI} = 114$ processes for faster results.
  With the use of emulators, such minima can be computed with $\mathcal{O}(1)$ MPI workers.

  <p align="center">
  <img width="750" height="750" alt="Screenshot 2025-08-13 at 5 29 59 PM" src="https://github.com/user-attachments/assets/c43b8eea-ee2e-443d-a497-cb9b2dae2fc3" />
  </p>

- **Profile**: 

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE1.py \
            --root ./projects/lsst_y1/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 \
            --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN1.txt"

  - macOS (arm)
        
        mpirun -n 51 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE1.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN1.txt"
       
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE2.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC2.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE2" \
            --factor 3 --nstw 750 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2.txt"

  - macOS (arm)

        mpirun -n 114 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE2.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC2.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE2" \
            --factor 3 --nstw 750 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2.txt"
     
  The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ minimum value - factor*np.sqrt(np.diag(cov))
      end   value ~ minimum value + factor*np.sqrt(np.diag(cov))

  We advise ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, we recommend ${\rm factor} \ll 1$.

  The script of the plot below is provided at `projects/lsst_y1/scripts/EXAMPLE_PLOT_PROFILE1[2].py`

  Profile 1: `LSST-Y1 Cosmic Shear only`

  The Google Colab [Profile Likelihood (LSST-Y1 only)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSST_Y1)_Profile_Likelihoods.ipynb) can be used to reconstruct a similar figure.
  
  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_profile1" src="https://github.com/user-attachments/assets/11f0f0dd-23e6-4875-bd8e-afbb11ac4e48" />
  </p>

  Profile 2: `Planck CMB (l < 396) + SN + BAO + LSST-Y1 Cosmic Shear`

  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_profile2" src="https://github.com/user-attachments/assets/cd041f96-dc42-426e-84a7-2d6498218b5f" />
  </p>

# Running Hybrid Cosmolike-ML emulators <a name="cobaya_base_code_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

Our main line of research involves emulators that simulate the entire Cosmolike data vectors, and each project (LSST, Roman, DES) contains its own README with emulator examples. The speed of such emulators is incredible, especially when GPUs are available, and our emulators do take advantage of the CPU-GPU integration on Apple MX chips. For example, the average timing of lsst-y1 cosmic shear data vector emulation is around 0.005s ($\sim$ 200828 evaluations in $\sim$ 850.5 seconds) on a macOS M2 Pro.

While the data vector emulators are incredibly fast, there is an intermediate approach that emulates only the Boltzmann outputs (comoving distance, linear and nonlinear matter power spectrum). This hybrid-ML case can offer greater flexibility, especially in the initial phases of a research project, as changes to the modeling of nuisance parameters or to the assumed galaxy distributions do not require retraining of the network. 

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). The required flags on `set_installation_options.sh` are similar to what we showed in the previous emulator section.

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4, the ideal setting for hybrid examples.

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1
  
 **Step :three:** Run `cobaya-run` on the first emulator example, following the commands below (here we only provide lsst-y1 examples).

- **One model evaluation**:

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)
    
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_MCMC1.yaml -r
    
Details on the matter power spectrum emulator designs will be presented in the [emulator_code](https://github.com/SBU-COSMOLIKE/emulators_code) repository. Basically, we apply standard neural network techniques to generalize the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) to new models, extended ranges, or higher precision. Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

> [!NOTE] 
> Users can decide not to correct the *syren-new* formula for the linear power spectrum (flag in the yaml). Although we have not conducted extensive studies of the caveats of the syren-new approximation, it appears sufficient for w0waCDM forecasts when combined with the Euclid Emulator to compute the nonlinear boost.
>
> For back-of-the-envelope LCDM calculations (e.g., to test cosmolike features), users can also choose not to correct the *syren-Halofit* formula for the LCDM nonlinear boost (see figure below). In this case, the overhead on top of cosmolike computations is minimum, at the order of $0.01$ seconds on a macOS M2Pro laptop. 
>
> <p align="center">
>  <span style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
>    <img width="450" alt="compare_emul_hemul" src="https://github.com/user-attachments/assets/7ada9b3b-db01-499f-8170-a2db5ef90636" />
>    <img width="450" alt="Screenshot 2026-01-06 at 2 13 06 AM" src="https://github.com/user-attachments/assets/d19f0900-cebc-41ca-9028-1d4e0bd40cc9" />
>  </span>
> </p>

# Running Fisher <a name="lsst_examples_fisher"></a>

  The Jupyter notebook `projects/lsst_y1/EXAMPLE_EVALUATE1.ipynb` provides code to compute the Fisher Matrix (Cosmic Shear). Hard priors are implemented in GetDist.

  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_fisher" src="https://github.com/user-attachments/assets/4f4e5775-6522-4eca-af82-4d1a10f4b1c8" />
  </p>



# Unit tests

The `tests/` folder holds 18 pass/fail tests and advisory checks.
The pass/fail tests compare the $\chi^2$ of cosmic shear, 3x2pt, and
2x2pt (each in NLA and TATT) against stored references within 0.2,
re-evaluate each fiducial as the 10th of 10 cosmologies in a row
under `OMP_NUM_THREADS=4` to catch state leaks and OpenMP races, and
check the python FAST-PT implementation of the TATT terms against
its own stored reference (printing the FASTPT-minus-cfastpt
difference). Advisory files measure the hybrid emulated pipelines
against the exact physics and the numerical error of the default
accuracy settings. Everything evaluated comes from the tests' own snapshot under
`tests/frozen/`, pinned by a SHA-256 manifest. From the `Cocoa/` folder, with the cocoa
environment active and `start_cocoa.sh` sourced:

    python -m pytest ./projects/lsst_y1/tests

`tests/README.md` describes every test and how to refresh the snapshot.

# Minimum accuracy parameters

The accuracy checks (`tests/test_accuracy.py`) measured, at the $\chi^2$
minimum on the 3x2pt configuration, changing one accuracy
parameter at a time:

| setting                            | raised to | $\Delta\chi^2$ |
|------------------------------------|-----------|-----------:|
| cosmolike `accuracyboost`          | 3         |    +0.0002 |
| cosmolike `accuracyboost` (stress) | 5         |    +0.0016 |
| cosmolike `integration_accuracy`   | 10        |     -0.001 |
| cosmolike `lmax`                   | 200000    |    +0.0002 |
| `kmax_boltzmann` + CAMB `kmax`     | 40 + 50   |    -0.0003 |
| CAMB `AccuracyBoost`               | 2         |     +0.012 |
| CAMB `k_per_logint`                | 50        |    +0.0001 |

`accuracyboost` refines a nested z grid in the power-spectrum tables:
every coarser grid's nodes are a subset of every finer grid's, so a
higher boost tightens the same interpolation instead of moving the
nodes (the construction is commented in
`likelihood/_cosmolike_prototype_base.py`). The raised-at-once checks
(comparing the default `accuracyboost: 1` against 3) stay within
-0.015 to +0.009 across cosmic shear, 3x2pt, and 2x2pt, in NLA and
TATT. All of this sits far below
the 0.2 comfort level: the default accuracy settings of this project
are adequate, and no default was changed.

When several settings move the $\chi^2$ in any project, raise cosmolike
`accuracyboost` first (cheap), then camb `k_per_logint`, and only
then camb `AccuracyBoost` (expensive at run time, and able to
masquerade for the cheap settings). `kmax_boltzmann` and camb `kmax`
are one physical cutoff seen from two sides and move together.
