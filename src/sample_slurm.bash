#!/bin/sh
# Specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=4
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory
#SBATCH --mem=14G
# Run for five minutes
#SBATCH --time=30:00
# Name job
#SBATCH --job-name=SbatchJob
# Name output file
#SBATCH --output=%x_%j.out
# Request email to be sent at begin and end, and if fails
#SBATCH --mail-type=ALL
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
python3 -m "print(input())"