from demLib.tilegrid import Tile
from demLib import File
import sys

"""
Script to read edges of the tiles and store them in a text file
that has the same name as the tile file but with a .edge extension.
The file structure is as follows:
The text file contains a dictionary -
                        {
                            'r' : [right_edge_vals, right_edge_loc],
                            'b' : [bottom_edge_vals, bottom_edge_loc],
                            'l' : [left_edge_vals, left_edge_loc],
                            't' : [top_edge_vals, top_edge_loc],
                            'n' : no_data_value
                        }

Usage example:

python get_edges.py tile_list_file.txt /tmp/edge_folder .tif


Here the tile_list_file.txt file contains list of DEM raster GeoTiff file paths,
the edege_folder is the output folder for all the edge text files, and .tif is the 
extension of the raster DEM files

"""


if __name__ == '__main__':

    script, tile_list_file, edge_folder, pattern = sys.argv

    # read text line by line from the rank list file
    filelines = File(tile_list_file).file_lines()

    # create a list of Tile objects
    tile_list = list(Tile(filename=filename.strip(), get_array=False) for filename in filelines)

    # iterate thru all the tile objects in the list
    for tile in tile_list:

        # read image as numpy array
        tile.read_array()

        # extract tile edges
        tile.extract_edges()

        # name of edge file
        edge_file = ''.join(tile.filename.split('.')[:-1] + ['.edge'])

        # write edges to text file
        tile.write_edges(edge_file)

        tile.array = None
