#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=6
#SBATCH --time=8:00:00
#SBATCH --mem=300gb
#SBATCH --job-name=pix_cc2
#SBATCH --export=NONE

source $HOME/.bashrc
conda activate pixie

python ./pixie.cell_clustering_cli.py --function cell_clustering_step2 --base_dir /gpfs/bwfor/work/ws/hd_gr294-mi_lunaphore_mcmicro/pixie_subset --pixel_output_dir lunaphore_pixel_masked_0.05_wseg_pixel_output_dir --cell_cluster_prefix cell_clustering_0.05  --pixel_cluster_col pixel_meta_cluster_rename
