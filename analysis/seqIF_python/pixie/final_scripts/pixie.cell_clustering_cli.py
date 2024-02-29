# import required packages
import argparse
import json
import os
from datetime import datetime as dt
import time

import feather
import matplotlib.pyplot as plt
import pandas as pd
from alpineer import load_utils
from matplotlib import rc_file_defaults

from ark.phenotyping import (cell_cluster_utils, cell_meta_clustering,
                             cell_som_clustering, weighted_channel_comp)
from ark.utils import data_utils, example_dataset, plot_utils
from ark.utils.metacluster_remap_gui import (MetaClusterGui,
                                             colormap_helper,
                                             metaclusterdata_from_files)
from alpineer import io_utils, load_utils


def cell_clustering_step1(base_dir, pixel_output_dir, cell_cluster_prefix,  pixel_cluster_col):
    start_time = time.time()
    print("Started pixie cell clustering CLI...")

    # base_dir = "/gpfs/bwfor/work/ws/hd_gr294-mi_lunaphore_mcmicro/pixie_subset"

    # define the name of the folder containing the pixel cluster data
    # pixel_output_dir = 'lunaphore_pixel_masked_0.05_pixel_output_dir'

    # explicitly set cell_cluster_prefix to override datetime default
    # cell_cluster_prefix = "cell_clustering_0.05subset"

    # define the type of pixel cluster to aggregate on
    # pixel_cluster_col = 'pixel_meta_cluster_rename'

    # define the name of the cell clustering params file
    cell_clustering_params_name = 'cell_clustering_params.json'

    # define the name of the directory with the extracted image data
    tiff_dir = os.path.join(base_dir, "image_data")

    # load the params
    with open(os.path.join(base_dir, "pixie", pixel_output_dir, cell_clustering_params_name)) as fh:
        cell_clustering_params = json.load(fh)

    # assign the params to variables
    fovs = cell_clustering_params['fovs']
    channels = cell_clustering_params['channels']
    segmentation_dir = cell_clustering_params['segmentation_dir']
    seg_suffix = cell_clustering_params['seg_suffix']
    pixel_data_dir = cell_clustering_params['pixel_data_dir']
    pc_chan_avg_som_cluster_name = cell_clustering_params['pc_chan_avg_som_cluster_name']
    pc_chan_avg_meta_cluster_name = cell_clustering_params['pc_chan_avg_meta_cluster_name']

    # define the cell table path
    cell_table_path = os.path.join(base_dir, 'segmentation',
                                   'cell_table', 'cell_table_size_normalized.csv')

    if cell_cluster_prefix is None:
        cell_cluster_prefix = dt.now().strftime('%Y-%m-%dT%H:%M:%S')

    # define the base output cell folder
    cell_output_dir = '%s_cell_output_dir' % cell_cluster_prefix
    if not os.path.exists(os.path.join(base_dir, "pixie", cell_output_dir)):
        os.mkdir(os.path.join(base_dir, "pixie", cell_output_dir))

    # define the paths to cell clustering files, explicitly set the variables to use custom names
    cell_som_weights_name = os.path.join("pixie", cell_output_dir, 'cell_som_weights.feather')
    cluster_counts_name = os.path.join("pixie", cell_output_dir, 'cluster_counts.feather')
    cluster_counts_size_norm_name = os.path.join(
        "pixie", cell_output_dir, 'cluster_counts_size_norm.feather')
    weighted_cell_channel_name = os.path.join(
        "pixie", cell_output_dir, 'weighted_cell_channel.feather')
    cell_som_cluster_count_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_som_cluster_count_avg.csv')
    cell_meta_cluster_count_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_count_avg.csv')
    cell_som_cluster_channel_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_som_cluster_channel_avg.csv')
    cell_meta_cluster_channel_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_channel_avg.csv')
    cell_meta_cluster_remap_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_mapping.csv')

    print("Started processing...")

    # depending on which pixel_cluster_col is selected, choose the pixel channel average table accordingly
    if pixel_cluster_col == 'pixel_som_cluster':
        pc_chan_avg_name = pc_chan_avg_som_cluster_name
    elif pixel_cluster_col == 'pixel_meta_cluster_rename':
        pc_chan_avg_name = pc_chan_avg_meta_cluster_name

    if os.path.exists(os.path.join(base_dir, cluster_counts_name)) and os.path.exists(os.path.join(base_dir, cluster_counts_size_norm_name)):
        # load the data if it exists
        cluster_counts = feather.read_dataframe(os.path.join(base_dir, cluster_counts_name))
        cluster_counts_size_norm = feather.read_dataframe(
            os.path.join(base_dir, cluster_counts_size_norm_name))
    else:
        # generate the preprocessed data
        cluster_counts, cluster_counts_size_norm = cell_cluster_utils.create_c2pc_data(
            fovs, os.path.join(base_dir, pixel_data_dir), cell_table_path, pixel_cluster_col
        )

        # write both unnormalized and normalized input data for reference
        feather.write_dataframe(
            cluster_counts,
            os.path.join(base_dir, cluster_counts_name),
            compression='uncompressed'
        )
        feather.write_dataframe(
            cluster_counts_size_norm,
            os.path.join(base_dir, cluster_counts_size_norm_name),
            compression='uncompressed'
        )

    # define the count columns found in cluster_counts_norm
    cell_som_cluster_cols = cluster_counts_size_norm.filter(
        regex=f'{pixel_cluster_col}.*'
    ).columns.values

    # depending on which pixel_cluster_col is selected, choose the pixel channel average table accordingly
    if pixel_cluster_col == 'pixel_som_cluster':
        pc_chan_avg_name = pc_chan_avg_som_cluster_name
    elif pixel_cluster_col == 'pixel_meta_cluster_rename':
        pc_chan_avg_name = pc_chan_avg_meta_cluster_name

    if not os.path.exists(os.path.join(base_dir, weighted_cell_channel_name)):
        # generate the weighted cell channel expression data
        pixel_channel_avg = pd.read_csv(os.path.join(base_dir, pc_chan_avg_name))
        weighted_cell_channel = weighted_channel_comp.compute_p2c_weighted_channel_avg(
            pixel_channel_avg,
            channels,
            cluster_counts,
            fovs=fovs,
            pixel_cluster_col=pixel_cluster_col
        )

        # write the data to weighted_cell_channel_name
        feather.write_dataframe(
            weighted_cell_channel,
            os.path.join(base_dir, weighted_cell_channel_name),
            compression='uncompressed'
        )

    print("Building SOM...")

    # create the cell SOM weights
    cell_pysom = cell_som_clustering.train_cell_som(
        fovs,
        base_dir,
        cell_table_path=cell_table_path,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        som_weights_name=cell_som_weights_name,
        num_passes=1,
        seed=42
    )

    # use cell SOM weights to assign cell clusters
    cluster_counts_size_norm = cell_som_clustering.cluster_cells(
        base_dir,
        cell_pysom,
        cell_som_cluster_cols=cell_som_cluster_cols
    )

    # intermediate saving of cell data with SOM labels assigned
    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    # generate the SOM cluster summary files
    cell_som_clustering.generate_som_avg_files(
        base_dir,
        cluster_counts_size_norm,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name
    )

    max_k = 20
    cap = 3

    # run hierarchical clustering using average count of pixel clusters per cell SOM cluster
    cell_cc, cluster_counts_size_norm = cell_meta_clustering.cell_consensus_cluster(
        base_dir,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name,
        max_k=max_k,
        cap=cap
    )

    # intermediate saving of cell data with SOM and meta labels assigned
    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    # generate the meta cluster summary files
    cell_meta_clustering.generate_meta_avg_files(
        base_dir,
        cell_cc,
        cell_som_cluster_cols=cell_som_cluster_cols,
        cell_som_input_data=cluster_counts_size_norm,
        cell_som_expr_col_avg_name=cell_som_cluster_count_avg_name,
        cell_meta_expr_col_avg_name=cell_meta_cluster_count_avg_name
    )

    # generate weighted channel summary files
    weighted_channel_comp.generate_wc_avg_files(
        fovs,
        channels,
        base_dir,
        cell_cc,
        cell_som_input_data=cluster_counts_size_norm,
        weighted_cell_channel_name=weighted_cell_channel_name,
        cell_som_cluster_channel_avg_name=cell_som_cluster_channel_avg_name,
        cell_meta_cluster_channel_avg_name=cell_meta_cluster_channel_avg_name
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time / 60:.2f} minutes")


def cell_clustering_step2(base_dir, pixel_output_dir, cell_cluster_prefix, pixel_cluster_col):
    start_time = time.time()
    print("Started pixie cell clustering CLI step2...")

    # explicitly set cell_cluster_prefix to override datetime default
    # cell_cluster_prefix = "cell_clustering_0.05subset"

    # define the name of the cell clustering params file
    cell_clustering_params_name = 'cell_clustering_params.json'

    # define the name of the directory with the extracted image data
    tiff_dir = os.path.join(base_dir, "image_data")

    # define the cell table path
    cell_table_path = os.path.join(base_dir, 'segmentation',
                                   'cell_table', 'cell_table_size_normalized.csv')

    # define the base output cell folder
    cell_output_dir = '%s_cell_output_dir' % cell_cluster_prefix
    if not os.path.exists(os.path.join(base_dir, "pixie", cell_output_dir)):
        os.mkdir(os.path.join(base_dir, "pixie", cell_output_dir))

    # define the paths to cell clustering files, explicitly set the variables to use custom names
    cell_som_weights_name = os.path.join("pixie", cell_output_dir, 'cell_som_weights.feather')
    cluster_counts_name = os.path.join("pixie", cell_output_dir, 'cluster_counts.feather')
    cluster_counts_size_norm_name = os.path.join(
        "pixie", cell_output_dir, 'cluster_counts_size_norm.feather')
    weighted_cell_channel_name = os.path.join(
        "pixie", cell_output_dir, 'weighted_cell_channel.feather')
    cell_som_cluster_count_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_som_cluster_count_avg.csv')
    cell_meta_cluster_count_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_count_avg.csv')
    cell_som_cluster_channel_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_som_cluster_channel_avg.csv')
    cell_meta_cluster_channel_avg_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_channel_avg.csv')
    cell_meta_cluster_remap_name = os.path.join(
        "pixie", cell_output_dir, 'cell_meta_cluster_mapping.csv')

    # load the params
    with open(os.path.join(base_dir, "pixie", pixel_output_dir, cell_clustering_params_name)) as fh:
        cell_clustering_params = json.load(fh)

    # assign the params to variables
    fovs = cell_clustering_params['fovs']
    channels = cell_clustering_params['channels']
    segmentation_dir = cell_clustering_params['segmentation_dir']
    seg_suffix = cell_clustering_params['seg_suffix']
    pixel_data_dir = cell_clustering_params['pixel_data_dir']
    pc_chan_avg_som_cluster_name = cell_clustering_params['pc_chan_avg_som_cluster_name']
    pc_chan_avg_meta_cluster_name = cell_clustering_params['pc_chan_avg_meta_cluster_name']

    cluster_counts_size_norm = feather.read_dataframe(
        os.path.join(base_dir, cluster_counts_size_norm_name))

    rc_file_defaults()
    cell_mcd = metaclusterdata_from_files(
        os.path.join(base_dir, cell_som_cluster_count_avg_name),
        cluster_type='cell',
        prefix_trim=pixel_cluster_col + '_'
    )
    cell_mcd.output_mapping_filename = os.path.join(base_dir, cell_meta_cluster_remap_name)

    # define the count columns found in cluster_counts_norm
    cell_som_cluster_cols = cluster_counts_size_norm.filter(
        regex=f'{pixel_cluster_col}.*'
    ).columns.values

    # rename the meta cluster values in the cell dataset
    cluster_counts_size_norm = cell_meta_clustering.apply_cell_meta_cluster_remapping(
        base_dir,
        cluster_counts_size_norm,
        cell_meta_cluster_remap_name
    )

    # intermediate saving of cell data with SOM, raw meta, and renamed meta labels assigned
    feather.write_dataframe(
        cluster_counts_size_norm,
        os.path.join(base_dir, cluster_counts_size_norm_name),
        compression='uncompressed'
    )

    print("Generating avg count files...")

    # recompute the mean column expression per meta cluster and apply these new names to the SOM cluster average data
    cell_meta_clustering.generate_remap_avg_count_files(
        base_dir,
        cluster_counts_size_norm,
        cell_meta_cluster_remap_name,
        cell_som_cluster_cols,
        cell_som_cluster_count_avg_name,
        cell_meta_cluster_count_avg_name,
    )

    # recompute the mean weighted channel expression per meta cluster and apply these new names to the SOM channel average data
    weighted_channel_comp.generate_remap_avg_wc_files(
        fovs,
        channels,
        base_dir,
        cluster_counts_size_norm,
        cell_meta_cluster_remap_name,
        weighted_cell_channel_name,
        cell_som_cluster_channel_avg_name,
        cell_meta_cluster_channel_avg_name
    )

    # raw_cmap, renamed_cmap = colormap_helper.generate_meta_cluster_colormap_dict(
    #     cell_mcd.output_mapping_filename,
    #     cell_mcg.im_cl.cmap,
    #     cluster_type='cell'
    # )

    # weighted_channel_comp.generate_weighted_channel_avg_heatmap(
    # os.path.join(base_dir, cell_som_cluster_channel_avg_name),
    # 'cell_som_cluster',
    # channels,
    # raw_cmap,
    # renamed_cmap
    # )

    # weighted_channel_comp.generate_weighted_channel_avg_heatmap(
    # os.path.join(base_dir, cell_meta_cluster_channel_avg_name),
    # 'cell_meta_cluster_rename',
    # channels,
    # raw_cmap,
    # renamed_cmap
    # )

    # select fovs to display
    #subset_cell_fovs = ['fov0', 'fov1']  # Replace with all fovs
    subset_cell_fovs = io_utils.list_folders(tiff_dir) ## This should get the names of all FOVs
    

    # generate and save the cell cluster masks for each fov in subset_cell_fovs
    data_utils.generate_and_save_cell_cluster_masks(
        fovs=subset_cell_fovs,
        save_dir=os.path.join(base_dir, "pixie", cell_output_dir),
        seg_dir=os.path.join(base_dir, segmentation_dir),
        cell_data=cluster_counts_size_norm,
        seg_suffix=seg_suffix,
        sub_dir='cell_masks',
        name_suffix='_cell_mask'
    )

    cell_cluster_utils.add_consensus_labels_cell_table(
        base_dir, cell_table_path, cluster_counts_size_norm
    )

    print("Creating mantis directory...")
    plot_utils.create_mantis_dir(
        fovs=subset_cell_fovs,
        mantis_project_path=os.path.join(base_dir, "pixie", cell_output_dir, "mantis"),
        img_data_path=tiff_dir,
        mask_output_dir=os.path.join(base_dir, "pixie", cell_output_dir, "cell_masks"),
        mapping=os.path.join(base_dir, cell_meta_cluster_remap_name),
        seg_dir=os.path.join(base_dir, segmentation_dir),
        cluster_type='cell',
        mask_suffix="_cell_mask",
        seg_suffix_name=seg_suffix
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time / 60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cell clustering pipeline')
    parser.add_argument('--function', type=str, help='function to run')
    parser.add_argument('--base_dir', type=str, help='base directory for data')
    parser.add_argument('--pixel_output_dir', type=str, help='output directory for data')
    parser.add_argument('--cell_cluster_prefix', type=str, help='prefix for cell clusters')
    parser.add_argument('--pixel_cluster_col', type=str, help='column name for pixel clusters')
    args = parser.parse_args()

    if args.function == 'cell_clustering_step1':
        cell_clustering_step1(args.base_dir, args.pixel_output_dir,
                              args.cell_cluster_prefix, args.pixel_cluster_col)
    elif args.function == 'cell_clustering_step2':
        cell_clustering_step2(args.base_dir, args.pixel_output_dir,
                              args.cell_cluster_prefix, args.pixel_cluster_col)
