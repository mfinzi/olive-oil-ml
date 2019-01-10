#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=12000  # memory in Mb
#SBATCH -J pbe_delta # Job name
#SBATCH -o mf.o%j # Name of stdout output file (%j expands to jobId)
#SBATCH -e mf.o%j # Name of stderr output file (%j expands to jobId)
#SBATCH -t 24:00:00  # time requested in hour:minute:second
#SBATCH --partition=default_gpu --gres=gpu:1 # Which queue it should run on.

python apply.py