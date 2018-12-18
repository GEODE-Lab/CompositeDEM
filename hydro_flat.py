import sys
import osr
from common import Raster, Vector


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
    raster = Raster(filename=raster_name)
    raster_spref = osr.SpatialReference()
    res = raster_spref.ImportFromWkt(raster.metadata['spref'])

    hydro_vector = Vector(filename=hydro_file)

    raster_bounds = raster.get_bounds()
    print('Raster bounds vector:')
    print(raster_bounds)

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
    '''
    example:
    hydro_flat.py "/temp/dem/astgdem_nad83_cgrid1_lzw.tif" "/temp/dem/astgdem_nad83_cgrid1_lzw_hyd_flat.tif"
    "/temp/dem/lakes/lakes.shp" 10
    '''

    raster_file, raster_out_file, hydro_shape_file = sys.argv[1:4]

    if len(sys.argv) > 4:
        percentile = sys.argv[4]
    else:
        percentile = 10  # default

    if len(sys.argv) > 5:
        min_pixels = sys.argv[5]
    else:
        min_pixels = 25  # default

    print('\nHydro-flattening...\n')
    main(raster_file, raster_out_file, hydro_shape_file, percentile, min_pixels)
    print('\n----------------------------------------------\n Done!\n')
