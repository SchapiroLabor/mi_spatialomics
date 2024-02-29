#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=6
#SBATCH --time=16:00:00
#SBATCH --mem=600gb
#SBATCH --job-name=pix_s1_m
#SBATCH --export=NONE

source $HOME/.bashrc
conda activate pixie

python /home/hd/hd_hd/hd_gr294/projects/spatial_MI/pixie/masked/notebook_to_cli.py --function pre_mapping --base_dir /gpfs/bwfor/work/ws/hd_gr294-mi_lunaphore_mcmicro/pixie_subset --channels ANKRD1 aSMA CCR2 CD31 CD45 CD68 MPO PDGFRa TNNT2 TREM2 --pixel_cluster_prefix "lunaphore_pixel_masked_0.05_wseg" --batch_size 8 --subset_proportion 0.05 --segmentation_dir "segmentation" --seg_suffix "_whole_cell.tiff"