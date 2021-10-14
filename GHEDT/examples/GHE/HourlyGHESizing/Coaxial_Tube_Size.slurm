#!/bin/sh

#SBATCH --job-name=CTSize
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jack.cook@okstate.edu
#SBATCH -p bigmem
#SBATCH --time 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=poisson_%A-%a.out    # Standard output and error log
# The following needs edited before each submission
#SBATCH --array=0-5

pwd;hostname;date

# Activate local installation of anaconda
conda_activate=/home/jackcook/anaconda3/bin/activate
conda_envs_dir=/home/jackcook/anaconda3/envs
conda_env=GLHE

source ${conda_activate} ${conda_env}

# pygfunction-v2.0 needs python3 and some specific packages to python3.7

# Declare array of volumetric flow rates
V_flow_rates = (7.8 15.6 31.2 46.8 62.4 78)

python Coaxial_Tube_HOTS_GHE.py V_flow_rates[${SLURM_ARRAY_TASK_ID}]