#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00
#SBATCH --mem=300gb
#SBATCH --job-name=pix_s2_m
#SBATCH --export=NONE

source $HOME/.bashrc
conda activate pixie

python /home/hd/hd_hd/hd_gr294/projects/spatial_MI/pixie/masked/notebook_to_cli.py --function post_mapping --base_dir /gpfs/bwfor/work/ws/hd_gr294-mi_lunaphore_mcmicro/pixie_subset --channels ANKRD1 aSMA CCR2 CD31 CD45 CD68 MPO PDGFRa TNNT2 TREM2 --pixel_cluster_prefix "lunaphore_pixel_masked_0.05_wseg" --batch_size 8 --segmentation_dir "segmentation" --seg_suffix "_whole_cell.tiff"
