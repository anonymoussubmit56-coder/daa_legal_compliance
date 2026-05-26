#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=rhxv40@durham.ac.uk
#SBATCH --mail-type=ALL

source /nobackup/rhxv40/biggerPhi/env_phi/bin/activate
python main_phi_normal.py
