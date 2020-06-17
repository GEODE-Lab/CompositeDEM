from demLib import Raster, Vector, Common
from demLib.parser import HydroParserMulti
import sys

"""
Script to extract stats as attributes from noisy lake surfaces across rasters 
spanned by large lakes using a boundary shapefile of the lakes.
The computed stats include mean, std deviation, percentiles from 0 - 100 in steps of 5

usage: python multi_tile_hydro_attr.py [-h] [--buffer BUFFER] [--max_ntiles MAX_NTILES] [--verbose]
                                multi_lake_tiles tile_file out_shpfile raster_file_dir

positional arguments:
  multi_lake_tiles      Shapefile containing lake polygons spanning multiple tiles
  tile_file             Shapefile containing tile footprints (unbuffered)
  out_shpfile           Output shapefile with multi-tile lakes and the tiles they cross as attributes
  raster_file_dir       Folder containing raster tiles with filenames corresponding to tile grid ID
                        with a .tif extension

optional arguments:
  -h, --help            show this help message and exit
  --buffer BUFFER, -b BUFFER
                        Distance in projection coords to buffer the tile boundary (default: 2000 for
                        ABoVE projection)
  --max_ntiles MAX_NTILES, -m MAX_NTILES
                        Maximum number of tiles spanned by a lake, number of tile-name attributes
                        (default: 69)
  --verbose, -v         Display verbosity (default: False)
"""


def main(multi_lake_tiles,
         tile_file,
         out_shpfile,
         raster_file_dir,
         buffer,
         max_ntiles,
         verbose):

    stats = ['mean', 'std_dev'] + list('pctl_{}'.format(str(i*5)) for i in range(0, 21))

    multi_vec = Vector(multi_lake_tiles)
    tile_vec = Vector(tile_file)

    if verbose:
        sys.stdout.write("Multi lakes file: {}\n".format(multi_lake_tiles))
        sys.stdout.write("Tile shapefile: {}\n".format(tile_file))
        sys.stdout.write("Output shapefile: {}\n".format(out_shpfile))
        sys.stdout.write("Raster file folder: {}\n".format(raster_file_dir))
        sys.stdout.write("Tile buffer: {}\n".format(str(buffer)))
        sys.stdout.write("Maximum number of overlapping tiles: {}\n".format(str(max_ntiles)))
        sys.stdout.write("Stats to calculate: {}\n\n".format(' '.join(stats)))

    vec_outside_buffer = Vector(geom_type=3,
                                spref_str=multi_vec.spref_str)

    vec_outside_buffer.add_field('filename', 'string', width=35)
    vec_outside_buffer.add_field('orig_id', 'string', width=50)
    vec_outside_buffer.add_field('n_tiles', 'int')

    for tile_idx, tile_num in enumerate(range(max_ntiles)):
        vec_outside_buffer.add_field('tile{}'.format(str(tile_idx + 1)),
                                     'string',
                                     width=10)

    vec_tile_dict = {}

    for vec_idx in range(multi_vec.nfeat):

        lake_geom = Vector.get_osgeo_geom(multi_vec.wktlist[vec_idx])

        tile_names = list(multi_vec.attributes[vec_idx]['tile{}'.format(tile_idx + 1)]
                          for tile_idx in range(multi_vec.attributes['n_tiles']))

        tile_idxs = []
        for tile_name in tile_names:

            tile_idx = next((tile_idx for tile_idx in range(len(tile_vec.nfeat))
                             if tile_vec.attributes["grid_id"] == tile_name), None)
            if tile_idx is not None:
                tile_idxs.append(tile_idx)

        crossing_tile_idxs = []
        if len(tile_idxs) > 0:
            for tile_idx in tile_idxs:
                tile_geom = Vector.get_osgeo_geom(tile_vec.wktlist[tile_idx])

                tile_geom_buffer = tile_geom.Buffer(buffer)

                if tile_geom_buffer.Crosses(lake_geom):
                    crossing_tile_idxs.append(tile_idx)

        if len(crossing_tile_idxs) > 0:
            if verbose:
                sys.stdout.write('Lake {} : Tiles - {}\n'.format(str(multi_vec.attributes['orig_id']),
                                                                 ' '.join([tile_vec.attributes[tile_idx]['grid_id']
                                                                          for tile_idx in crossing_tile_idxs])))
            vec_tile_dict[vec_idx] = crossing_tile_idxs

    if verbose:
        sys.stdout.write('Number of Multi-tile lakes: {}\n'.format(len(vec_tile_dict)))
        sys.stdout.write('Extracting multi-tile values....')

    out_vec = Vector(spref_str=multi_vec.spref_str,
                     geom_type=3,
                     in_memory=True)

    out_vec.add_field('orig_id', 'string', width=50)

    for stat in stats:
        out_vec.add_field(stat, 'float', precision=8)

    for vec_idx, tile_idx_list in vec_tile_dict.items():
        vec_geom = Vector.get_osgeo_geom(multi_vec.wktlist[vec_idx])

        pixel_list = []

        for tile_idx in tile_idx_list:
            tile_geom = Vector.get_osgeo_geom(tile_vec.wktlist[tile_idx])
            vec_tile_intrsctn = vec_geom.Intersection(tile_geom)

            raster_tile = raster_file_dir + tile_vec.attributes[tile_idx]['grid_id'] + '.tif'

            vector_tile = Vector(spref_str=multi_vec.spref_str,
                                 geom_type=3,
                                 in_memory=True)
            vector_tile.add_feat(vec_tile_intrsctn)

            tile = Raster(raster_tile,
                          get_array=True)

            pixel_list += tile.vector_extract(vec_tile_intrsctn,
                                              return_values=True)

        vec_stats = dict(zip(stats, Common.get_stats(pixel_list, stats)))
        vec_attr = {'orig_id': multi_vec.attributes['orig_id'],
                    'fid': vec_idx}

        vec_attr.update(vec_stats)

        out_vec.add_feat(vec_geom, attr=vec_attr)

    out_vec.write_vector(out_shpfile)


if __name__ == '__main__':

    args = HydroParserMulti().parser.parse_args()

    main(args.multi_lake_tiles,
         args.tile_file,
         args.out_shpfile,
         args.raster_file_dir,
         args.buffer,
         args.max_ntiles,
         args.verbose)

    if args.verbose:
        sys.stdout.write('\n----------------------------------------------\n Done!\n')






