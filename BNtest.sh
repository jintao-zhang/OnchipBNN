#!/bin/sh
# note you will want to edit following lines,
# especially if you want to use more resources
# The following is for testing and allocates
# 2 minutes only, on 1 node, 1 GPU

#SBATCH --job-name=BN_test.py
#SBATCH --output=BNtest.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH --workdir=./
#SBATCH --gres=gpu:1

# can remove following line if it is in your ~/.bash_profile etc
export MODULEPATH=/tigress/PNI/modulefiles:$MODULEPATH

module load anaconda
module load cudatoolkit
module load intel-mkl
source activate /tigress/jintao/theano
python ./Train-time/mnist.py

