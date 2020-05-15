import sys
import osr
import argparse
from demLib.spatial import Raster, Vector

'''
Script to flatten noisy lake surfaces in a raster DEM (.tif) using a boundary shapefile of the lakes. 

Script syntax: 

python hydro_flat.py [DEM raster file] [DEM raster output file] [lakes shape file] [percentile clip] [min pixels]

percentile clip: Percentile value for final elevation of flat surface
min pixels: Minimum number of raster pixels inside a feature below which no flattening is desired

example:
hydro_flat.py "astgdem.tif" "astgdem_hydflat.tif" "lakes.shp" --percentile 10 --min_pixels 25
'''


def main(raster_name,
         out_raster_name,
         hydro_file,
         pctl=10,
         min_pixels=25):

    """
    Main function to run hydro flattening
    :param raster_name: Raster filename with full path
    :param out_raster_name: The output file to write the final raster
    :param hydro_file: Shapefile of water body boundaries
    :param pctl: Percentile value to substitute (default: 10)
    :param min_pixels: Number of minimum pixels for extraction (default: 25)
    :return: None
    """

    # initialize objects
    raster = Raster(filename=raster_name,
                    get_array=True)
    raster_spref = osr.SpatialReference()
    res = raster_spref.ImportFromWkt(raster.metadata['spref'])

    hydro_vector = Vector(filename=hydro_file)

    raster_bounds = raster.get_bounds(bounds_vector=True)
    print('Raster bounds vector: {}'.format(raster_bounds))

    # find intersecting tile features
    hydro_vector_reproj = hydro_vector.reproject(destination_spatial_ref=raster_spref,
                                                 _return=True)
    print(hydro_vector_reproj)

    intersect_vector = hydro_vector_reproj.get_intersecting_vector(raster_bounds)
    print(intersect_vector)

    # replace values by percentile
    result = raster.vector_extract(intersect_vector,
                                   pctl=pctl,
                                   replace=True,
                                   min_pixels=min_pixels)

    # write to disk
    raster.write_raster(outfile=out_raster_name)

    raster = tile_vector = tiles_vector = intersect_vector = None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script for hydro-flattening water bodies.")

    parser.add_argument("raster_infile",
                        type=str,
                        help="Input raster file name")
    parser.add_argument("raster_outfile",
                        type=str,
                        help="Output raster file name")
    parser.add_argument("hydro_shpfile",
                        type=str,
                        help="Shapefile of water bodies")

    parser.add_argument("--percentile",
                        default=10,
                        type=int,
                        help="Percentile for flattened output (default: 10)")
    parser.add_argument("--min_pixels",
                        default=25,
                        type=int,
                        help="Minimum pixels for flattening (default: 25)")

    args = parser.parse_args()

    sys.stdout.write('\nHydro-flattening - {}\n'.format(args.raster_infile))

    main(args.raster_infile,
         args.raster_outfile,
         args.hydro_shpfile,
         args.percentile,
         args.min_pixels)

    sys.stdout.write('\n----------------------------------------------\n Done!\n')
