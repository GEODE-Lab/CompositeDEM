from demLib.tilegrid import Tile
from demLib import File, Common
import sys

"""
Script to apply delta surface fill. This script uses the following:
1) Text file with full path names of raster files (or VRTs) in the order of their rank for filling
2) Path name of the output filled file

Example of script use:

> python delta.py /tmp/rank_list_file_h001_v001.txt /tmp/output_tile_h001_v001.tif

Example text in rank list text file:
 
/tmp/thisIsVeryFirstDEM_tileH001V001.tif
/tmp/TotallySecondDEMFile_tile_h001_v001.tif
/tmp/tremendouslyBestly_third_tile_h001__v001.tif
/tmp/OhMyGod__ItsFourth_tile_h__001_V_001.tif

"""


if __name__ == '__main__':

    script, rank_list_file, outfile = sys.argv

    Common.cprint('===============\nRank list file: {}\n'.format(rank_list_file))

    # read text line by line from the rank list file
    filelines = File(rank_list_file).file_lines()

    # make a list of files
    rank_list = list(filename.strip() for filename in filelines)

    Common.cprint('=== DEM files ===')
    for filename in rank_list:
        Common.cprint(filename)
    Common.cprint('\n=== Delta method ===')

    # DEM tile with first rank and its edge file name
    tile = Tile(filename=rank_list[0])
    edge_file = ''.join(rank_list[0].split('.')[:-1] + ['.edge'])

    # loop over the rest of tiles
    for indx in range(len(rank_list) - 1):

        # extract next tile
        nxt_rank_tile = Tile(filename=rank_list[indx + 1])

        Common.cprint('- {} ---- {} -'.format(rank_list[indx],
                                              rank_list[indx + 1]))

        # subtract previous tile from the next tile
        # voids from the tiles are copied to the difference tile
        tile_diff = tile - nxt_rank_tile

        # update the difference tile using edge file if available
        # if no edge file is available this step will warn and not do anything
        tile_diff.update_array(edge_file)

        # fill the voids in the difference tile
        # if the previous step did not load any tile edges, then
        # the void interpolation will not fill voids on the edges
        tile_diff.fill()

        # add the next tile to the difference tile,
        # after all the voids are filled
        tile = tile_diff + nxt_rank_tile

        # copy voids from the next tile to the output
        tile.copy_voids(nxt_rank_tile)

        # this edge file will be used in the next loop
        edge_file = ''.join(rank_list[indx + 1].split('.')[:-1] + ['.edge'])

    Common.cprint('\n=== Delta complete ===\n\nWriting Raster {}\n'.format(outfile))

    # finally, write the tile to a raster file
    tile.write_raster(outfile)
    Common.cprint('Done!')
