import argparse


class HydroParser(object):
    """
    Parser object for hydro_flat.py
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Script to flatten noisy lake surfaces in a raster DEM"
                                                          " (.tif) using a boundary shapefile of the lakes. ")

        self.parser.add_argument("raster_infile",
                                 type=str,
                                 help="Input raster file name")
        self.parser.add_argument("raster_outfile",
                                 type=str,
                                 help="Output raster file name")
        self.parser.add_argument("hydro_shpfile",
                                 type=str,
                                 help="Shapefile of water bodies")

        self.parser.add_argument("--percentile", "-p",
                                 default=10,
                                 type=int,
                                 help="Percentile value for final elevation of flat surface (default: 10)")
        self.parser.add_argument("--min_pixels", "-minp",
                                 default=25,
                                 type=int,
                                 help="Minimum number of raster pixels inside a feature below which " + \
                                      "no flattening is desired (default: 25)")
        self.parser.add_argument("--verbose", "-v",
                                 action="store_true",
                                 help='Display verbosity (default: False)')

    def __repr__(self):
        return "<Parser object for hydro_flat.py>"


class HydroParserMulti(object):
    """
    Parser object for hydro_flat.py
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Script to flatten noisy lake surfaces across rasters " +
                                                          "spanned by large lakes" +
                                                          " using a boundary shapefile of the lakes. ")

        self.parser.add_argument("multi_lake_tiles",
                                 type=str,
                                 help="Shapefile containing lake polygons spanning multiple tiles")
        self.parser.add_argument("tile_file",
                                 type=str,
                                 help="Shapefile containing tile footprints (unbuffered)")
        self.parser.add_argument("out_shpfile",
                                 type=str,
                                 help="Output shapefile with multi-tile lakes and the tiles they cross as attributes")
        self.parser.add_argument("raster_file_dir",
                                 type=str,
                                 help="Folder containing raster tiles with filenames corresponding to " +
                                      "tile grid ID with a .tif extension")

        self.parser.add_argument("--buffer", "-b",
                                 default=2000,
                                 type=float,
                                 help="Distance in projection coords to buffer the tile boundary " +
                                      "(default: 2000 for ABoVE projection)")
        self.parser.add_argument("--max_ntiles", "-m",
                                 default=69,
                                 type=int,
                                 help="Maximum number of tiles spanned by a lake, number of tile-name attributes " +
                                      "(default: 69)")

        self.parser.add_argument("--verbose", "-v",
                                 action="store_true",
                                 help='Display verbosity (default: False)')

    def __repr__(self):
        return "<Parser object for multi_tile_hydro_flat.py>"

