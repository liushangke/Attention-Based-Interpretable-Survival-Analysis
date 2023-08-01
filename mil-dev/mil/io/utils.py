import histomics_stream as hs
import os
import tensorflow as tf


# slide_metadata keys for import into other io modules.
slide_keys = [
    "filename",
    "level",
    "read_magnification",
    "returned_magnification",
    "scan_magnification",
    "slide_name",
    "slide_group",
    "number_pixel_rows_for_chunk",
    "number_pixel_columns_for_chunk",
    "number_pixel_rows_for_tile",
    "number_pixel_columns_for_tile",
    "number_pixel_rows_for_slide",
    "number_pixel_columns_for_slide",
    "number_tile_rows_for_slide",
    "number_tile_columns_for_slide",
]

# tile_metadata keys for import into other io modules.
tile_keys = ["chunk_left", "chunk_top", "tile_left", "tile_top", "slide_index"]


import histomics_stream as hs


def study(
    paths, t=(224, 224), overlap=(0, 0), chunk=(1792, 1792), target=20, source="exact"
):
    """Generate a histomics stream study for one of more slides from a single subject.

    Parameters
    ----------
    paths : string, tuple(string), or list
        Paths to the whole-slide images and optionally masks to include in the histomics_stream study.
        If string, a single study is generated from a single whole-slide image w/o masking. If
        tuple, a single study is generated from a single (slide, mask) pair. Lists inputs are used to
        generate multiple studies from multiple slides or (slide, mask) pairs. Strings and tuples can
        be mixed in list inputs if some slides lack masks.
    t : tuple(int, int)
        The height and width (pixels) of the tiles used for analysis at target magnification. Default
        value is (224, 224).
    overlap : tuple(int, int)
        The vertical and horizontal overlap (pixels) between tiles. Default value is (0, 0).
    chunk : tuple(int, int)
        The size of a region to retrieve from the slide at the target magnification. histomics_stream
        groups the reading of tiles into multi-tile chunks to minimize overhead. Default value is
        (1792, 1792).
    target : float
        The target magnification to return tiles at. If this magnification is not available, histomics_stream
        will read the next highest magnification and resize to obtain the desired magnification. Default
        value is 20 to analyze at 20X objective magnification.
    source : string
        A histomics_stream parameter defining read and resizing behavior. Default value exact returns the
        exact magnification requested, using resizing if necessary. See histomics_stream documentation
        for more details.
    Returns
    -------
    study : object
        A histomics_stream study object containing the slides defined in paths, and analysis
        plan defined by tile size, tile overlap, and magnification/reading parameters.
    """

    # helper functions to process tuple, string inputs
    def string_input(path):
        file = os.path.split(path)[1]
        name = os.path.splitext(file)[0]
        return file, name

    def tuple_input(path):
        file = os.path.split(path[0])[1]
        name = os.path.splitext(file)[0]
        return file, name

    # wrap string or tuple inputs in list and check arguments
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, tuple):
        paths = [paths]
    elif isinstance(paths, list):
        for path in paths:
            if not (isinstance(path, str) or isinstance(path, tuple)):
                raise ValueError(
                    "Argument 'paths' must be a string, tuple of (string, string) or list of these elements."
                )
    else:
        raise ValueError(
            "Argument 'paths' must be a string, tuple of (string, string) or list of these elements."
        )

    # extract names from lists
    names = []
    for path in paths:
        if isinstance(path, str):
            file = os.path.split(path)[1]
        elif isinstance(path, tuple):
            file = os.path.split(path[0])[1]
        names.append(os.path.splitext(file)[0])

    # fill basic study parameters
    study = {"version": "version-1"}
    study["number_pixel_rows_for_tile"] = t[0]
    study["number_pixel_columns_for_tile"] = t[1]
    slides = study["slides"] = {}

    # add slides to study
    for i, (name, path) in enumerate(zip(names, paths)):
        if isinstance(path, tuple):
            filename = path[0]
        else:
            filename = path
        slide_name = os.path.splitext(os.path.split(filename)[1])[0]
        slides[name] = {
            "filename": filename,
            "slide_name": slide_name,
            "slide_group": name,
            "number_pixel_rows_for_chunk": chunk[0],
            "number_pixel_columns_for_chunk": chunk[1],
        }

    # apply settings to each slide
    for name, path in zip(names, paths):
        # generate resolution setting function
        find_resolution_for_slide = hs.configure.FindResolutionForSlide(
            study, target_magnification=target, magnification_source=source
        )

        # generate gridding function
        if isinstance(path, tuple):
            tiles_by_grid_and_mask = hs.configure.TilesByGridAndMask(
                study,
                number_pixel_overlap_rows_for_tile=overlap[0],
                number_pixel_overlap_columns_for_tile=overlap[1],
                mask_filename=path[1],
            )
        else:
            tiles_by_grid_and_mask = hs.configure.TilesByGridAndMask(
                study,
                number_pixel_overlap_rows_for_tile=overlap[0],
                number_pixel_overlap_columns_for_tile=overlap[1],
            )

        # apply functions
        find_resolution_for_slide(study["slides"][name])
        tiles_by_grid_and_mask(study["slides"][name])

    return study


def inference(model, study, batch=128, prefetch=2):
    """Applies a model to a histomics_stream study, returning the extracted
    features and tile_info.
    Parameters
    ----------
    model : tf.keras.Model
        The model to use for extracting features from tiles.
    study : object
        A histomics_stream study object containing the slides defined in paths, and analysis
        plan defined by tile size, tile overlap, and magnification/reading parameters.
    batch : int, default 128
        The number of tiles to process in a batch.
    prefetch : int, default 2
        The number of batches to prefetch.
    Returns
    -------
    features : float
        A two-dimensional feature tensor produced by histomics_stream with instances in rows.
    tile_info : dict
        A dictionary of file, magnification, and position data for each tile produced
        by histomics_stream.
    """

    # generate batched tf dataset
    create_tf_dataset = hs.tensorflow.CreateTensorFlowDataset()
    tiles = create_tf_dataset(study)
    tiles = tiles.map(lambda x, y, z: ((x[0], x[1]), 0.0))
    tiles = tiles.batch(batch)
    tiles = tiles.prefetch(prefetch)

    # generate features
    features, tile_info = model.predict(tiles)

    return features, tile_info
