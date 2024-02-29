import argparse
import json
import os
from datetime import datetime as dt
import time

import matplotlib.pyplot as plt
from alpineer import io_utils, load_utils
from matplotlib import rc_file_defaults

from ark.phenotyping import (pixel_cluster_utils, pixel_meta_clustering,
                             pixel_som_clustering, pixie_preprocessing)
from ark.utils import data_utils, example_dataset, plot_utils
from ark.utils.metacluster_remap_gui import (MetaClusterGui,
                                             colormap_helper,
                                             metaclusterdata_from_files)


def function_pre_remapping(base_dir, channels, blur_factor=2, pixie_seg_dir=None, batch_size=32, pixel_cluster_prefix='example', filter_channel='CD163', nuclear_exclude=True, seed=42, cluster_type='pixel', subset_pixel_fovs=['fov0', 'fov1'], metacluster_colors=None, subset_proportion=1.0,
                           segmentation_dir=None, seg_suffix=None, multiprocess=False, blurred_channels=None, smooth_vals=6):
    start_time = time.time()
    print("Started pixie CLI...")

    tiff_dir = os.path.join(base_dir, "image_data")
    img_sub_folder = None

    if segmentation_dir is not None:
        pixie_seg_dir = os.path.join(base_dir, segmentation_dir)
    else:
        pixie_seg_dir = None

    # either get all fovs in the folder...
    fovs = io_utils.list_folders(tiff_dir)

    multiprocess = False

    if pixel_cluster_prefix is None:
        pixel_cluster_prefix = dt.now().strftime('%Y-%m-%dT%H:%M:%S')
    # define the output directory using the specified pixel cluster prefix
    pixel_output_dir = os.path.join("pixie", "%s_pixel_output_dir" % pixel_cluster_prefix)
    if not os.path.exists(os.path.join(base_dir, pixel_output_dir)):
        os.makedirs(os.path.join(base_dir, pixel_output_dir))

    # define the preprocessed pixel data folders
    pixel_data_dir = os.path.join(pixel_output_dir, 'pixel_mat_data')
    pixel_subset_dir = os.path.join(pixel_output_dir, 'pixel_mat_subset')
    norm_vals_name = os.path.join(pixel_output_dir, 'channel_norm_post_rowsum.feather')
    # set an optional list of markers for additional blurring
    # blurred_channels = ["ECAD"]
    # smooth_vals = 6
#     blurred_channels = None

#     pixel_cluster_utils.smooth_channels(
#         fovs=fovs,
#         tiff_dir=tiff_dir,
#         img_sub_folder=img_sub_folder,
#         channels=blurred_channels,
#         smooth_vals=smooth_vals,
#     )

    # pixel_cluster_utils.filter_with_nuclear_mask(
    #     fovs=fovs,
    #     tiff_dir=tiff_dir,
    #     seg_dir=os.path.join(base_dir, segmentation_dir),
    #     channel=filter_channel,
    #     nuc_seg_suffix="_nuclear.tiff",
    #     img_sub_folder=img_sub_folder,
    #     exclude=nuclear_exclude
    # )

    # run pixel data preprocessing
    print("Preprocessing pixels matrix...")
    pixie_preprocessing.create_pixel_matrix(
        fovs,
        channels,
        base_dir,
        tiff_dir,
        pixie_seg_dir,
        img_sub_folder=img_sub_folder,
        seg_suffix=seg_suffix,
        pixel_output_dir=pixel_output_dir,
        data_dir=pixel_data_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        blur_factor=blur_factor,
        subset_proportion=subset_proportion,
        multiprocess=multiprocess,
        batch_size=batch_size
    )
    pixel_som_weights_name = os.path.join(pixel_output_dir, 'pixel_som_weights.feather')
    pc_chan_avg_som_cluster_name = os.path.join(
        pixel_output_dir, 'pixel_channel_avg_som_cluster.csv')
    pc_chan_avg_meta_cluster_name = os.path.join(
        pixel_output_dir, 'pixel_channel_avg_meta_cluster.csv')
    pixel_meta_cluster_remap_name = os.path.join(
        pixel_output_dir, 'pixel_meta_cluster_mapping.csv')

    print("Started training SOM...")
    # create the pixel SOM weights
    pixel_pysom = pixel_som_clustering.train_pixel_som(
        fovs,
        channels,
        base_dir,
        subset_dir=pixel_subset_dir,
        norm_vals_name=norm_vals_name,
        som_weights_name=pixel_som_weights_name,
        num_passes=1,
        seed=42
    )
    # use pixel SOM weights to assign pixel clusters
    pixel_som_clustering.cluster_pixels(
        fovs,
        channels,
        base_dir,
        pixel_pysom,
        data_dir=pixel_data_dir,
        multiprocess=multiprocess,
        batch_size=batch_size
    )

    # generate the SOM cluster summary files
    pixel_som_clustering.generate_som_avg_files(
        fovs,
        channels,
        base_dir,
        pixel_pysom,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name
    )
    max_k = 20
    cap = 3

    # run hierarchical clustering using average pixel SOM cluster expression
    pixel_cc = pixel_meta_clustering.pixel_consensus_cluster(
        fovs,
        channels,
        base_dir,
        max_k=max_k,
        cap=cap,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        multiprocess=multiprocess,
        batch_size=batch_size
    )

    # generate the meta cluster summary files
    pixel_meta_clustering.generate_meta_avg_files(
        fovs,
        channels,
        base_dir,
        pixel_cc,
        data_dir=pixel_data_dir,
        pc_chan_avg_som_cluster_name=pc_chan_avg_som_cluster_name,
        pc_chan_avg_meta_cluster_name=pc_chan_avg_meta_cluster_name
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time / 60:.2f} minutes")


def function_post_remapping(base_dir, channels, pixie_seg_dir=None, batch_size=5, pixel_cluster_prefix="example",
                            pc_chan_avg_meta_cluster_name=None, cluster_type="pixel", subset_pixel_fovs=['fov0', 'fov1'], metacluster_colors=None, seg_suffix=None,
                            segmentation_dir=None):
    start_time = time.time()
    print("Started pixie CLI...")
    # recompute the mean channel expression per meta cluster and apply these new names to the SOM cluster average data

    tiff_dir = os.path.join(base_dir, "image_data")
    fovs = io_utils.list_folders(tiff_dir)
    
    # define the preprocessed pixel data folders
    pixel_output_dir = os.path.join("pixie", "%s_pixel_output_dir" % pixel_cluster_prefix)
    pixel_data_dir = os.path.join(pixel_output_dir, 'pixel_mat_data')  # Need to add
    pc_chan_avg_som_cluster_name = os.path.join(
        pixel_output_dir, 'pixel_channel_avg_som_cluster.csv')
    pc_chan_avg_meta_cluster_name = os.path.join(
        pixel_output_dir, 'pixel_channel_avg_meta_cluster.csv')
    pixel_meta_cluster_remap_name = os.path.join(
        pixel_output_dir, 'pixel_meta_cluster_mapping.csv')
    
    ## Assign new clusters to pixel metaclusters (breaks when run in Jupyter notebook)
    pixel_meta_clustering.apply_pixel_meta_cluster_remapping(
        fovs,
        channels,
        base_dir,
        pixel_data_dir,
        pixel_meta_cluster_remap_name,
        multiprocess=False,
        batch_size=batch_size)


    pixel_meta_clustering.generate_remap_avg_files(
        fovs,
        channels,
        base_dir,
        pixel_data_dir,
        pixel_meta_cluster_remap_name,
        pc_chan_avg_som_cluster_name,
        pc_chan_avg_meta_cluster_name
    )

    pixel_mcd = metaclusterdata_from_files(
        os.path.join(base_dir, pc_chan_avg_som_cluster_name),
        cluster_type='pixel'
    )
    pixel_mcd.output_mapping_filename = os.path.join(base_dir, pixel_meta_cluster_remap_name)
    pixel_mcg = MetaClusterGui(pixel_mcd, width=17)

    raw_cmap, _ = colormap_helper.generate_meta_cluster_colormap_dict(
        pixel_mcd.output_mapping_filename,
        pixel_mcg.im_cl.cmap
    )
    # select fovs to display
    subset_pixel_fovs = fovs
    # define the path to the channel file
    img_sub_folder = None  # Need to remap ##
    if img_sub_folder is None:
        chan_file = os.path.join(
            io_utils.list_files(os.path.join(tiff_dir, fovs[0]), substrs=['.tif'])[0]
        )
    else:
        chan_file = os.path.join(
            img_sub_folder, io_utils.list_files(os.path.join(
                tiff_dir, fovs[0], img_sub_folder), substrs=['.tif'])[0]
        )

    print("Generating pixel masks...")
    # generate and save the pixel cluster masks for each fov in subset_pixel_fovs
    data_utils.generate_and_save_pixel_cluster_masks(
        fovs=subset_pixel_fovs,
        base_dir=base_dir,
        save_dir=os.path.join(base_dir, pixel_output_dir),
        tiff_dir=tiff_dir,
        chan_file=chan_file,
        pixel_data_dir=pixel_data_dir,
        pixel_cluster_col='pixel_meta_cluster',
        sub_dir='pixel_masks',
        name_suffix='_pixel_mask',
    )

    # Currently non-working function
    # plot_utils.save_colored_masks(
    #     fovs=subset_pixel_fovs,
    #     mask_dir=os.path.join(base_dir, pixel_output_dir, "pixel_masks"),
    #     save_dir=os.path.join(base_dir, pixel_output_dir, "pixel_mask_colored"),
    #     cluster_id_to_name_path=os.path.join(base_dir, pixel_meta_cluster_remap_name),
    #     metacluster_colors=raw_cmap,
    #     cluster_type="pixel"
    # )

    for pixel_fov in subset_pixel_fovs:
        pixel_cluster_mask = load_utils.load_imgs_from_dir(
            data_dir=os.path.join(base_dir, pixel_output_dir, "pixel_masks"),
            files=[pixel_fov + "_pixel_mask.tiff"],
            trim_suffix="_pixel_mask",
            match_substring="_pixel_mask",
            xr_dim_name="pixel_mask",
            xr_channel_names=None,
        )

        plot_utils.plot_pixel_cell_cluster_overlay(
            pixel_cluster_mask,
            [pixel_fov],
            os.path.join(base_dir, pixel_meta_cluster_remap_name),
            metacluster_colors=raw_cmap
        )

    print("Saving pixel clustering parameters...")
    # define the params dict
    cell_clustering_params = {
        'fovs': io_utils.remove_file_extensions(io_utils.list_files(os.path.join(base_dir, pixel_data_dir), substrs='.feather')),
        'channels': channels,
        'segmentation_dir': segmentation_dir,
        'seg_suffix': seg_suffix,
        'pixel_data_dir': pixel_data_dir,
        'pc_chan_avg_som_cluster_name': pc_chan_avg_som_cluster_name,
        'pc_chan_avg_meta_cluster_name': pc_chan_avg_meta_cluster_name
    }
    with open(os.path.join(base_dir, pixel_output_dir, 'cell_clustering_params.json'), 'w') as fh:
        json.dump(cell_clustering_params, fh)

    print("Saving Mantis directory...")

    if segmentation_dir is not None:
        pixie_seg_dir = os.path.join(base_dir, segmentation_dir)

        plot_utils.create_mantis_dir(
            fovs=subset_pixel_fovs,
            mantis_project_path=os.path.join(base_dir, pixel_output_dir, "mantis"),
            img_data_path=tiff_dir,
            mask_output_dir=os.path.join(base_dir, pixel_output_dir, "pixel_masks"),
            mapping=os.path.join(base_dir, pixel_meta_cluster_remap_name),
            seg_dir=pixie_seg_dir,
            mask_suffix="_pixel_mask",
            seg_suffix_name=seg_suffix
        )
    else:
        pixie_seg_dir = None
        plot_utils.create_mantis_dir(
            fovs=subset_pixel_fovs,
            mantis_project_path=os.path.join(base_dir, pixel_output_dir, "mantis"),
            img_data_path=tiff_dir,
            mask_output_dir=os.path.join(base_dir, pixel_output_dir, "pixel_masks"),
            mapping=os.path.join(base_dir, pixel_meta_cluster_remap_name)
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time / 60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI for Jupyter Notebook translation.')
    parser.add_argument('--function', type=str,
                        choices=['pre_mapping', 'post_mapping'], required=True, help='Function to run.')
    parser.add_argument('--base_dir', default=None, type=str,
                        help='Base directory for pixie.')
    parser.add_argument('--channels', nargs='+', help='List of channels.')
    parser.add_argument('--blur_factor', default=2, type=int, help='Blur factor.')
    parser.add_argument('--subset_proportion', default=0.1,
                        type=float, help='Proportion of FOVs to use.')
    # Add arguments for user-defined variables
    parser.add_argument('--pixie_seg_dir', default=None,
                        type=type(None), help='Value for pixie_seg_dir.')
    parser.add_argument('--batch_size', default=5, type=type(int),
                        help='Value for batch_size.')
    parser.add_argument('--pixel_cluster_prefix', default="example",
                        type=type("example"), help='Value for pixel_cluster_prefix.')
    parser.add_argument('--filter_channel', default='CD163',
                        type=type('CD163'), help='Value for filter_channel.')
    parser.add_argument('--nuclear_exclude', default=True,
                        type=type(True), help='Value for nuclear_exclude.')
    parser.add_argument('--seed', default=42, type=type(42), help='Value for seed.')
    parser.add_argument('--pc_chan_avg_meta_cluster_name', default=None,
                        type=type(None), help='Value for pc_chan_avg_meta_cluster_name.')
    parser.add_argument('--cluster_type', default="pixel",
                        type=type("pixel"), help='Value for cluster_type.')
    parser.add_argument('--subset_pixel_fovs', default=['fov0', 'fov1'],
                        type=type(['fov0', 'fov1']), help='Value for subset_pixel_fovs.')
    parser.add_argument('--metacluster_colors', default=None,
                        type=type(None), help='Value for metacluster_colors.')
    # arser.add_argument('--cell_clustering_params', default={, type=type({), help='Value for cell_clustering_params.')
    parser.add_argument('--segmentation_dir', default=None, help='Segmentation directory.')
    parser.add_argument('--seg_suffix', default=None, help='Segmentation file suffix.')
    parser.add_argument('--multiprocess', action='store_true', help='Use multiprocessing.')
    parser.add_argument('--blurred_channels', nargs='+',
                        default=None, help='List of blurred channels.')
    parser.add_argument('--smooth_vals', nargs='+',
                        default=6, type=int, help='List of smoothing values.')

    args = parser.parse_args()

    if args.function == 'pre_mapping':
        function_pre_remapping(base_dir=args.base_dir, channels=args.channels, blur_factor=args.blur_factor, pixie_seg_dir=args.pixie_seg_dir, batch_size=args.batch_size, pixel_cluster_prefix=args.pixel_cluster_prefix, filter_channel=args.filter_channel, nuclear_exclude=args.nuclear_exclude, seed=args.seed, cluster_type=args.cluster_type,
                               subset_pixel_fovs=args.subset_pixel_fovs, metacluster_colors=args.metacluster_colors, subset_proportion=args.subset_proportion, segmentation_dir=args.segmentation_dir, multiprocess=args.multiprocess, blurred_channels=args.blurred_channels, smooth_vals=args.smooth_vals,
                               seg_suffix=args.seg_suffix)
    elif args.function == 'post_mapping':
        function_post_remapping(base_dir=args.base_dir, channels=args.channels, pixie_seg_dir=args.pixie_seg_dir, batch_size=args.batch_size, pixel_cluster_prefix=args.pixel_cluster_prefix,
                                pc_chan_avg_meta_cluster_name=args.pc_chan_avg_meta_cluster_name, cluster_type=args.cluster_type, subset_pixel_fovs=args.subset_pixel_fovs, metacluster_colors=args.metacluster_colors, seg_suffix=args.seg_suffix,
                                segmentation_dir=args.segmentation_dir)
