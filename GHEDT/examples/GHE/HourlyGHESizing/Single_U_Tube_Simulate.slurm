#!/bin/sh

# Check to make sure GHEDT is properly installed and the simualation is going to
# provide expected results prior to submitting jobs that will take nearly a day
# to complete

#SBATCH --job-name=SUSim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jack.cook@okstate.edu
#SBATCH -p bigmem
#SBATCH --time 5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=poisson_%A-%a.out    # Standard output and error log
# The following needs edited before each submission
#SBATCH --array=0

pwd;hostname;date

# Activate local installlation of anaconda
conda_activate=/home/jackcook/anaconda3/bin/activate
conda_envs_dir=/home/jackcook/anaconda3/envs
conda_env=GLHE

source ${conda_activate} ${conda_env}

# pygfunction-v2.0 needs python3 and some specific packages to python3.7

python simulate.py
