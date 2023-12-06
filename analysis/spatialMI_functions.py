import pandas as pd
import napari
from tifffile.tifffile import imread
import geopandas as gp
from shapely.geometry import Polygon
import distinctipy
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
import skimage.io as io
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def plot_pixie_maps(pixel_image, sample, mask_color, scale, screenshot_path, crop_out_path, roi=None):
    crop_pixel_map = crop_mask(roi, pixel_image)

    viewer = napari.Viewer()
    viewer.add_labels(crop_pixel_map, color=mask_color)
    # viewer.add_labels(boundaries, visible=True, name="mask",
    #                   opacity=1, color={0: 'transparent', 1: (1, 1, 1)})
    viewer.screenshot(path=screenshot_path,
                      scale=scale)

    crop_screenshot = crop_out_path+sample+".crop.png"
    crop_black_margins(screenshot_path, crop_screenshot)

    recolor_screenshot = crop_out_path+sample+".recolor.png"
    recolor_black_to_white(crop_screenshot, recolor_screenshot)


def recolor_black_to_white(img, recolor_screenshot: str) -> None:
    from PIL import Image
    import numpy as np

    img = io.imread(img)

    # Check the number of channels in the image
    num_channels = img.shape[2]

    # Step 2: Identify Black Pixels
    # Find the coordinates of black pixels; black is represented by [0, 0, 0]
    black_pixels = (img[:, :, :3] == [0, 0, 0]).all(axis=2)

    # Step 3: Replace Black Pixels with White
    # Set the RGB values of the identified black pixels to [255, 255, 255] (white)
    img[black_pixels, :3] = [255, 255, 255]

    # Step 4: Show the Resulting Image
    # Convert the numpy array back to an image
    output_image = Image.fromarray(img)

    output_image.save(recolor_screenshot)


def crop(roi_df, image):
    """
    Crop an image based on the coordinates of a region of interest (ROI).

    Parameters
    ----------
    roi_df : pandas.DataFrame
        A DataFrame containing the coordinates of the ROI. The last two columns
        should contain the y and x coordinates of the top-left and bottom-right
        corners of the ROI, respectively.
    image : numpy.ndarray
        The image to crop.

    Returns
    -------
    numpy.ndarray
        The cropped image.
    """
    # get last 2 columns, this is y, x
    bbox = roi_df.iloc[:, -2:].to_numpy()
    return image[:, int(bbox[0, 0]):int(bbox[2, 0]), int(bbox[0, 1]): int(bbox[2, 1])]


def crop_mask(roi_df, mask):
    # get last 2 columns, this is y, x
    bbox = roi_df.iloc[:, -2:].to_numpy()
    return mask[int(bbox[0, 0]):int(bbox[2, 0]), int(bbox[0, 1]): int(bbox[2, 1])]


def crop_coords(roi, points):
    """
    Adjust the coordinates of points based on the coordinates of a region of interest (ROI).

    Parameters
    ----------
    roi_df : pandas.DataFrame
        A DataFrame containing the coordinates of the ROI. The last two columns
        should contain the y and x coordinates of the top-left and bottom-right
        corners of the ROI, respectively.
    points : pandas.DataFrame
        A DataFrame containing the coordinates of the points to adjust. The x and y
        coordinates should be in separate columns.

    Returns
    -------
    pandas.DataFrame
        The adjusted DataFrame of points.
    """
    min_x = roi.iloc[0, -1]
    min_y = roi.iloc[0, -2]
    points.loc[:, 'x'] -= min_x
    points.loc[:, 'y'] -= min_y
    return points


def plot_layers_napari(image=None, image_channel_axis=0, image_channel_colors=None, points_data=None, genes_of_interest=None, roi=None, mask=None, color_palette=("green", "blue"), pt_size=1, output_path=".",
                       crop_margins=True, scale_bar=True, font_size=30, scale=1, scalebar_length=100, img_type="image",
                       roi_plot=None, box_edge_thickness=40, crop_out_path=None, sample=None, outline_mask=False, mask_color=None,
                       image_contrast_limits=None, channel_names=None):
    """
    Plot points on an image using Napari.

    Parameters
    ----------
    image : numpy.ndarray
        The image to plot the points on.
    points_data : pandas.DataFrame
        A DataFrame containing the coordinates of the points to plot. The x and y
        coordinates should be in separate columns, and there should be a column
        indicating the gene target of each point.
    genes_of_interest : list
        A list of gene targets to plot.
    roi : pandas.DataFrame, optional
        A DataFrame containing the coordinates of the region of interest (ROI).
        The last two columns should contain the y and x coordinates of the top-left
        and bottom-right corners of the ROI, respectively.
    mask : numpy.ndarray, optional
        A binary mask indicating which pixels in the image belong to the ROI.
    color_palette : tuple, optional
        A tuple of colors to use for the gene targets. The default is ("green", "blue").
    pt_size : int, optional
        The size of the points to plot. The default is 1.
    output_path : str, optional
        The path to save the output image. The default is ".".

    Returns
    -------
    None.

    Notes
    -----
    This function requires the Napari library to be installed. You can install it using pip:

    ```
    pip install napari
    ```

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from molkart.plotting import plot_points_napari
    >>> image = np.random.rand(100, 100)
    >>> points_data = pd.DataFrame({"x": np.random.randint(0, 100, size=100),
    ...                             "y": np.random.randint(0, 100, size=100),
    ...                             "gene": np.random.choice(["A", "B", "C"], size=100)})
    >>> genes_of_interest = ["A", "B"]
    >>> plot_points_napari(image, points_data, genes_of_interest, pt_size=5)
    """
    import napari
    from skimage.segmentation import find_boundaries

    if points_data is not None:
        # Just loading everything and converting to geopandas.
        points_data.columns = ['x', 'y', 'z', 'gene_target']
        points_data['gene_target'] = pd.Categorical(
            points_data['gene_target'], genes_of_interest)
        points_data = points_data[points_data.gene_target.isin(
            genes_of_interest)]
        gdf = gp.GeoDataFrame(
            points_data, geometry=gp.points_from_xy(
                points_data.x, points_data.y)
        )
    if roi is not None:
        # If roi, crop points, image and mask
        if image is not None:
            image_view = crop(roi, image)
            xmax, ymax = image_view.shape[2], image_view.shape[1]
        if mask is not None:
            mask_view = crop_mask(roi, mask)
            xmax, ymax = mask_view.shape[1], mask_view.shape[0]
        if points_data is not None:
            # Loading the polygon in x and y since that is how you gave the data, but napari saves in y, x order unless you rearranged dim order
            polygon = Polygon(roi.iloc[:, :-3:-1].to_numpy())
            poly_gpd = gp.GeoDataFrame(index=[0], geometry=[polygon])

            # Basically fastest way to get all points within a polygon.
            subset_points = gp.sjoin(gdf, poly_gpd, predicate='within')
            points_view = crop_coords(roi, subset_points)
            xmax, ymax = points_view["x"].max(), points_view["y"].max()

    else:
        # If no roi, just plot full view
        if image is not None:
            image_view = image
            xmax, ymax = image_view.shape[2], image_view.shape[1]
        if mask is not None:
            mask_view = mask
            xmax, ymax = mask_view.shape[1], mask_view.shape[0]
        if points_data is not None:
            points_view = points_data
            xmax, ymax = points_view["x"].max(), points_view["y"].max()

            points_view = points_view.sort_values(by='gene_target')
            points_view['cell_id'] = points_view.index
            # We use the gene target code, which is an integer as for the color cycle it is not accepted to have a string. However, with text we can still see the gene target
            points_props = {'cell_id': points_view['cell_id'].to_numpy(),
                            'gene_target': points_view['gene_target'].to_numpy()}

    viewer = napari.Viewer()
    if image is not None:
        if image.any():
            # viewer.add_image(image_view)
            viewer.add_image(
                image_view, channel_axis=image_channel_axis, colormap=image_channel_colors, contrast_limits=image_contrast_limits, name=channel_names)
    if mask is not None:
        if mask.any():
            if outline_mask == False:
                if (mask_color is not None):
                    viewer.add_labels(mask_view, color=mask_color)
                else:
                    viewer.add_labels(mask_view)

            else:
                boundaries_mask = find_boundaries(mask_view, mode='thick')
                viewer.add_image(
                    boundaries_mask, colormap='gray', contrast_limits=[0, 1])
    if points_data is not None:
        viewer.add_points(points_view[['y', 'x']].to_numpy(),
                          properties=points_props,
                          face_color='gene_target',
                          face_color_cycle=color_palette,
                          size=pt_size,
                          edge_width_is_relative=False)
    if roi_plot is not None:
        roi_array = [roi_plot.iloc[:, -2:].to_numpy()]
        viewer.add_shapes(roi_array, shape_type=['polygon'],
                          edge_color='white', edge_width=box_edge_thickness, face_color="transparent", opacity=1)
        # viewer.layers.save(output_path, plugin='napari-svg') ## Save as vector graphics
        # viewer.layers.save(output_path)

    viewer.screenshot(path=output_path,
                      scale=scale)

    # Crop black margins
    if crop_margins is not None:
        crop_screenshot = crop_out_path+sample+"."+img_type+".crop.png"
        crop_black_margins(output_path, crop_screenshot)

    # Add scalebar
    if scale_bar is not None:
        crop_scalebar = crop_out_path+sample+"."+img_type+".crop.scale.png"
        add_scalebar(crop_screenshot, ymax,
                     scalebar_length_um=scalebar_length,
                     corner="bottom right", image_with_scalebar_path=crop_scalebar, font_size=font_size)


def crop_black_margins(img, output_path: str) -> None:
    """
    This function takes a PIL Image as input, detects and crops black margins from the image, 
    and saves the cropped image to a specified path.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image from which to crop black margins.
    output_path : str
        The path at which to save the cropped image.

    Returns
    -------
    None
    """
    from PIL import Image
    import numpy as np

    image = Image.open(img)

    # Convert image to grayscale for simpler processing
    img_gray = image.convert('L')

    # Convert to numpy array for easier processing
    img_array = np.array(img_gray)

    # Detect left margin
    for margin_left in range(img_array.shape[1]):
        if np.any(img_array[:, margin_left] != 0):
            break

    # Detect right margin
    for margin_right in range(img_array.shape[1]-1, -1, -1):
        if np.any(img_array[:, margin_right] != 0):
            break

    # Detect top margin
    for margin_top in range(img_array.shape[0]):
        if np.any(img_array[margin_top, :] != 0):
            break

    # Detect bottom margin
    for margin_bottom in range(img_array.shape[0]-1, -1, -1):
        if np.any(img_array[margin_bottom, :] != 0):
            break

    # Crop the image
    img_cropped = image.crop(
        (margin_left, margin_top, margin_right, margin_bottom))

    # Save the cropped image
    img_cropped.save(output_path)


def add_scalebar(image_path, img_width_orig, scalebar_length_um, corner="bottom right", image_with_scalebar_path=".", font_size=None):
    """
    Add a scalebar to an image.

    Parameters:
    image_path (str): The path to the image file.
    pixel_resolution (float): The pixel resolution of the image in micrometers per pixel.
    scalebar_length_um (float): The desired length of the scalebar in micrometers.
    corner (str): The corner where the scalebar will be placed. Options are "bottom right", "bottom left", "top right", "top left".
    image_with_scalebar_path (str): The path to save the new image file with the scalebar. Default is the current directory.
    font_size (int): The font size of the scalebar text. If None, the font size will be automatically determined based on the scalebar length. Default is None.

    Returns:
    str: The path to the new image file with the scalebar.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os

    # Load the image
    image = Image.open(image_path)

    # Convert scalebar length from micrometers to pixels
    pixel_resolution = img_width_orig * 0.138 / image.width
    scalebar_length_px = int(scalebar_length_um / pixel_resolution)

    # Set scalebar parameters
    # Scalebar height proportional to its length
    scalebar_height = int(scalebar_length_px / 10)
    # Padding around scalebar and text
    scalebar_padding = int(scalebar_height / 4)
    # Padding between the box and the image border
    border_padding = int(scalebar_length_px * 0.10)

    # Determine font size based on scalebar length if not provided
    if font_size is None:
        # Font size proportional to scalebar height
        font_size = max(10, scalebar_height)

    # Set the font
    font = ImageFont.truetype("../../references/Arial.ttf", font_size)

    # Determine the size of the text
    # Get the mask of the text
    text_mask = font.getmask(f"{scalebar_length_um} µm")

    # Determine the size of the text
    text_width, text_height = text_mask.getbbox()[2:]

    # Set the scalebar position based on the chosen corner
    if corner == "bottom right":
        scalebar_position = (image.width - scalebar_length_px - scalebar_padding -
                             border_padding, image.height - scalebar_height - scalebar_padding - border_padding)
    elif corner == "bottom left":
        scalebar_position = (scalebar_padding + border_padding, image.height -
                             scalebar_height - scalebar_padding - border_padding)
    elif corner == "top right":
        scalebar_position = (image.width - scalebar_length_px - scalebar_padding -
                             border_padding, scalebar_padding + text_height + border_padding)
    elif corner == "top left":
        scalebar_position = (scalebar_padding + border_padding,
                             scalebar_padding + text_height + border_padding)
    else:
        raise ValueError(
            'The "corner" parameter should be one of the following: "bottom right", "bottom left", "top right", "top left"')

    # Position for the text to be centered on top of the scalebar
    text_position = (scalebar_position[0] + (scalebar_length_px - text_width) /
                     2, scalebar_position[1] - text_height - scalebar_padding)

    # Create a new transparent overlay for the scalebar background
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Draw a semi-transparent rectangle as the scalebar background
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        [
            (scalebar_position[0] - scalebar_padding,
             text_position[1] - scalebar_padding),
            (scalebar_position[0] + scalebar_length_px + scalebar_padding,
             scalebar_position[1] + scalebar_height + scalebar_padding)
        ],
        fill=(0, 0, 0, 128)  # RGBA
    )

    # Draw the scalebar
    draw.rectangle(
        [scalebar_position, (scalebar_position[0] + scalebar_length_px,
                             scalebar_position[1] + scalebar_height)],
        fill=(255, 255, 255, 255)  # RGBA
    )

    # Add text above the scalebar
    draw.text(text_position,
              f"{scalebar_length_um} µm", fill='white', font=font)

    # Overlay the scalebar onto the original image
    image_with_scalebar = Image.alpha_composite(image.convert('RGBA'), overlay)

    # Save the new image
    image_with_scalebar.save(image_with_scalebar_path)

    return image_with_scalebar_path
