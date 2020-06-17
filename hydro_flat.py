import sys
import osr
from demLib.spatial import Raster, Vector
from demLib.parser import HydroParser

'''
Script to flatten noisy lake surfaces in a raster DEM (.tif) using a boundary shapefile of the lakes.

usage: python hydro_flat.py [-h] [--multi_tile_file MULTI_TILE_FILE] [--percentile PERCENTILE] 
                            [--min_pixels MIN_PIXELS] [--verbose] raster_infile raster_outfile hydro_shpfile

positional arguments:
  raster_infile         Input raster file name
  raster_outfile        Output raster file name
  hydro_shpfile         Shapefile of water bodies

optional arguments:
  -h, --help            show this help message and exit
  --multi_tile_file MULTI_TILE_FILE, -mt MULTI_TILE_FILE
                        Shapefile with lakes spanning multiple tiles with stats as attributes (default: none)
  --percentile PERCENTILE, -p PERCENTILE
                        Percentile value for final elevation of flat surface (default: 10)
  --min_pixels MIN_PIXELS, -minp MIN_PIXELS
                        Minimum number of raster pixels inside a feature below which no flattening is desired (default: 25)
  --verbose, -v         Display verbosity (default: False)

'''


def main(raster_name,
         out_raster_name,
         hydro_file,
         multi_tile_file,
         pctl=10,
         min_pixels=25,
         verbose=False):
    """
    Main function to run hydro flattening
    :param raster_name: Raster filename with full path
    :param out_raster_name: The output file to write the final raster
    :param hydro_file: Shapefile of water body boundaries
    :param multi_tile_file: Output file from multi_tile_hydro_attr.py,
                            file must contain stat attributes, geometry_id, and geometry
    :param pctl: Percentile value to substitute (default: 10)
    :param min_pixels: Number of minimum pixels for extraction (default: 25)
    :param verbose: Display verbosity (Default: False)

    :return: None
    """

    # initialize objects
    raster = Raster(filename=raster_name,
                    get_array=True)
    raster_spref = osr.SpatialReference()
    res = raster_spref.ImportFromWkt(raster.metadata['spref'])

    hydro_vector = Vector(filename=hydro_file)

    raster_bounds = raster.get_bounds(bounds_vector=True)

    if multi_tile_file != 'none':
        multi_tile_vec = Vector(multi_tile_file)
    else:
        multi_tile_vec = None

    if verbose:
        sys.stdout.write('Raster bounds vector: {}\n'.format(raster_bounds))

    # find intersecting tile features
    hydro_vector_reproj = hydro_vector.reproject(destination_spatial_ref=raster_spref,
                                                 _return=True)
    if verbose:
        sys.stdout.write(hydro_vector_reproj)
        sys.stdout.write("\n")

    intersect_vector = hydro_vector_reproj.get_intersecting_vector(raster_bounds)

    if verbose:
        sys.stdout.write(intersect_vector)
        sys.stdout.write("\n")

    # replace values by percentile
    result = raster.vector_extract(intersect_vector,
                                   pctl=pctl,
                                   replace=True,
                                   min_pixels=min_pixels)

    if multi_tile_vec is not None:
        multi_tile_vec_attr_keys = list(multi_tile_vec.attributes)
        percentiles = sorted(list(int((key.split('_')[1]).strip()) for key in multi_tile_vec_attr_keys
                                  if 'pctl' in key))

        diff_from_val = list(abs(val - pctl) for val in percentiles)
        nearest_idx = diff_from_val.index(min(diff_from_val))

        pctl_attr = 'pctl_{}'.format(str(percentiles[nearest_idx]))

        geom_idx_list = []

        for multi_geom_idx in range(multi_tile_vec.nfeat):
            for intersect_geom_idx in range(intersect_vector.nfeat):
                if intersect_vector.attributes[intersect_geom_idx]['orig_id'] == \
                        multi_tile_vec.attributes[multi_geom_idx]['orig_id']:

                    geom_idx_list.append(multi_geom_idx)
                    break

        if verbose:
            sys.stdout.write("Found {} multi-tile geometries\n".format(str(len(geom_idx_list))))

        if len(geom_idx_list) > 0:
            for idx, geom_idx in enumerate(geom_idx_list):
                if verbose:
                    sys.stdout.write("Processing multi-tile geometry {} of {}\n".format(str(idx + 1),
                                                                                        str(len(geom_idx_list))))

                multi_vec = Vector(spref_str=raster_spref,
                                   geom_type=3,
                                   in_memory=True)
                multi_geom = Vector.get_osgeo_geom(multi_tile_vec.wktlist[geom_idx])
                multi_vec.add_feat(multi_geom)

                result = raster.vector_extract(multi_vec,
                                               pctl=pctl,
                                               min_pixels=min_pixels,
                                               replace=True,
                                               replace_val=multi_tile_vec.attributes[geom_idx][pctl_attr])

    # write to disk
    if verbose:
        sys.stdout.write('\nWriting raster file: {}\n'.format(out_raster_name))

    raster.write_raster(outfile=out_raster_name)

    raster = tile_vector = tiles_vector = intersect_vector = None


if __name__ == '__main__':

    args = HydroParser().parser.parse_args()

    if args.verbose:
        sys.stdout.write('\nHydro-flattening - {}\n'.format(args.raster_infile))

    main(args.raster_infile,
         args.raster_outfile,
         args.hydro_shpfile,
         args.multi_tile_file,
         args.percentile,
         args.min_pixels,
         args.verbose)

    if args.verbose:
        sys.stdout.write('\n----------------------------------------------\n Done!\n')
