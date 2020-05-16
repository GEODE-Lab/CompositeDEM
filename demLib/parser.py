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
                                 default=True,
                                 type=bool,
                                 help='Display verbosity (default: False)')

    def __repr__(self):
        return "<Parser object for hydro_flat.py>"
