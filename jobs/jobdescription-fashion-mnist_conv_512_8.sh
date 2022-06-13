#!/bin/bash

#SBATCH --job-name=UnifiedTransformer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.m.de.sousa.horta.osorio.laitenberger@student.rug.nl
#SBATCH --output=job-fashion-mnist_conv_512_8.log
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=64G

module purge
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4

source /data/$USER/.envs/python386-bachelors/bin/activate

python3 main.py --dataset fashion-mnist --image-embedding conv --embedding-dimension 512 --num-heads 8
