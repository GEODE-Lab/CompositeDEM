from scipy.interpolate import interp1d
from common import group_consecutive
from spatial import Raster
import numpy as np
import json


class Layer(object):

    """
    Class to store and manipulate Raster Tile as a numpy array layer
    """

    def __init__(self,
                 array=None,
                 edgex=None,
                 edgey=None,
                 nodata=None):
        """
        Instantiate class
        :param array: numpy array (1D)
        :param edgey: The two edges of the array (nrows x 2) columns:
                        1) left , 2) right
        :param edgex: The two edges of the array (2 x ncol) rows:
                       1) top , 2) bottom
        :param nodata:
        """

        self.array = array
        self.ncol = None
        self.nrows = None
        self.nodata = nodata
        self.edgex = edgex
        self.edgey = edgey

        if self.array is not None:
            if self.edgex is None:
                self.edgex = self.array[[0, -1], :]
            else:
                self.array[[0, -1], :] = self.edgex

            if self.edgey is None:
                self.edgey = self.array[:, [0, -1]]
            else:
                self.array[:, [0, -1]] = self.edgey

            self.ncol = array.shape[1]
            self.nrows = array.shape[0]

    def __repr__(self):
        return '<Layer of shape: {} x {}>'.format(str(self.nrows),
                                                  str(self.ncol))

    @staticmethod
    def find_blocks(arr,
                    nodata=None):
        """
        Method to find blocks of no data value in a 1-D array
        :param arr: Input 1-D array
        :param nodata: no data value
        :return: List of tuples [(pixel_loc_before_void, pixel_loc_after_void), ]
        """

        void_locs = np.msort(np.where(arr == nodata)[0])
        grouped_locs = group_consecutive(void_locs.tolist())

        blocks = []
        for group in grouped_locs:
            if len(group) > 0:
                blocks.append([group[0] - 1 if group[0] != 0 else -1,
                               group[-1] + 1 if group[-1] != arr.shape[0] else -1])

        return blocks

    @staticmethod
    def fill_voids(arr,
                   blocks=None):
        """
        Method to fill a void in a 1-D array
        if none of the block indices are negative
        :param arr: 1-D array
        :param blocks: List of tuples of block locations, output from find_blocks()
        :return: 1-D array
        """
        out_arr = arr.copy()

        if blocks is not None and len(blocks) > 0:
            for block in blocks:
                if not any([(indx < 0) for indx in block]):
                    y = out_arr[list(block)]
                    f = interp1d(block, y)
                    out_arr[np.arange(*block)] = f(np.arange(*block))

        return out_arr

    @staticmethod
    def fill_voids_1d(arr,
                      nodata):
        """
        Method to fill all voids in a 1D array
        :param arr: 1-D array
        :param nodata: No data value
        :return: 1-D array
        """
        return Layer.fill_voids(arr,
                                Layer.find_blocks(arr,
                                                  nodata))

    def fill(self):
        """
        Method to fill voids in 2D array by
            Filling voids in 1D array along x axis
            Filling voids in 1D array along y axis
            and taking the mean of two 2D arrays
            This methods leaves out voids at edges
        :return: 2D array
        """
        xfilled_arr = np.apply_along_axis(self.fill_voids_1d,
                                          1,
                                          self.array,
                                          self.nodata)

        yfilled_arr = np.apply_along_axis(self.fill_voids_1d,
                                          0,
                                          self.array,
                                          self.nodata)

        x_remain_voids = np.where(xfilled_arr == self.nodata)
        y_remain_voids = np.where(yfilled_arr == self.nodata)

        mean_arr = (xfilled_arr + yfilled_arr)/2.0

        mean_arr[x_remain_voids] = self.nodata
        mean_arr[y_remain_voids] = self.nodata

        return mean_arr


class Edge(object):
    """
    Class for storing and processing edge values for a Raster Tile
    """
    def __init__(self,
                 edge_dict=None,
                 filename=None,
                 nodata=None):
        """
        Instantiate Edge class
        :param edge_dict: Dictionary storing edge information in the following format:
                        {
                            'r' : [right_edge_vals, right_edge_loc],
                            'b' : [bottom_edge_vals, bottom_edge_loc],
                            'l' : [left_edge_vals, left_edge_loc],
                            't' : [top_edge_vals, top_edge_loc],
                            'n' : no_data_value
                        }
                Here, all the edge_vals and the edge_locs are 1D list objects.
                All elements should be present. Absence of even one element makes the
                dictionary incomplete and invalid. Unavailability of any element should be
                indicated by -1.

        :param filename: filename for edge_dict
        :param nodata: No data value
        """

        self.edges = edge_dict
        self.filename = filename
        self.nodata = nodata

    def __repr__(self):
        return '<Edge object {}>'.format('-MEM-' if self.filename is None else self.filename)

    def load(self,
             filename=None):
        """
        Method to load an edge dictionary from file
        :param filename: Name of file with edge information
        :return: None
        """

        if self.filename is None:
            if filename is not None:
                self.filename = filename
            else:
                raise ValueError('No valid file specified')

        with open(self.filename, 'r') as ef:
            self.edges = json.loads(ef.read())

        self.nodata = self.edges['n']

    def write(self,
              filename=None):
        """
        Method to write an edge dictionary to file
        :param filename: Name of file with edge information
        :return: None
        """
        if self.filename is None:
            if filename is not None:
                self.filename = filename
            else:
                raise ValueError('No valid file specified')

        with open(self.filename, 'w') as ef:
            ef.write(json.dumps(self.edges))

    @staticmethod
    def get_nearest_loc_val(arr,
                            nodata):
        """
        function to get nearest located non nodata value and
        its location for a 1D array
        :param arr: 1D array
        :param nodata: no data value
        :return: [array_begin_value, array_begin_value_location,
                  array_end_value, array_end_value_location]
        """
        nodata_loc = np.where(arr == nodata)[0]

        if nodata_loc.shape[0] > 0:
            blocks = Layer.find_blocks(arr,
                                       nodata)
            out_arr = []
            if blocks[0][0] != -1:
                out_arr += [arr[0], 0]
            elif blocks[0][1] != -1:
                out_arr += [arr[blocks[0][1]], blocks[0][1]]
            else:
                out_arr += [np.NAN, np.NAN]

            if blocks[-1][1] != -1:
                out_arr += [arr[-1], len(arr) - 1]
            elif blocks[-1][0] != -1:
                out_arr += [arr[blocks[-1][0]], blocks[-1][0]]
            else:
                out_arr += [np.NAN, np.NAN]

            return np.array(out_arr)

        else:
            return np.array([arr[0], 0, arr[-1], len(arr) - 1])

    def extract(self,
                array):
        """
        Method to extract edges from a tile
        :param array: 2D array
        :return: edge dict
        """
        # value and location array for left and right edges
        val_loc_arr_lr = np.apply_along_axis(lambda x: self.get_nearest_loc_val(x, self.nodata), 1, array)

        # value and location array for top and bottom edges
        val_loc_arr_tb = np.apply_along_axis(lambda x: self.get_nearest_loc_val(x, self.nodata), 0, array)

        edge_dict = dict()

        edge_dict['l'] = [val_loc_arr_lr[:, 0].tolist(), val_loc_arr_lr[:, 1].tolist()]
        edge_dict['r'] = [val_loc_arr_lr[:, 2].tolist(), val_loc_arr_lr[:, 3].tolist()]

        edge_dict['t'] = [val_loc_arr_tb[:, 0].tolist(), val_loc_arr_tb[:, 1].tolist()]
        edge_dict['b'] = [val_loc_arr_tb[:, 2].tolist(), val_loc_arr_tb[:, 3].tolist()]

        edge_dict['n'] = self.nodata

        self.edges = edge_dict


class Tile(Raster, Edge, Layer):

    """
    Class to amalgamate properties and methods of
    Raster, Edge, and Layer classes into a Tile object
    """

    def __init__(self,
                 filename,
                 edgefile=None,
                 nodata=None,
                 get_array=True):
        """
        Instantiate Tile class
        :param filename: Tile filepath (usually a .tif file)
        :param edgefile: filepath of .edge file
        :param nodata: No data value to use for voids
        """

        Raster.__init__(self,
                        filename,
                        get_array)

        Edge.__init__(self,
                      filename=edgefile,
                      nodata=nodata)

        if edgefile is not None:
            Edge.load(self,
                      filename=edgefile)
        elif get_array:
            Edge.extract(self,
                         self.array[0, :, :])

            Layer.__init__(self,
                           array=self.array,
                           nodata=nodata)

        # geometric properties of the Tile object:
        # bounds: (xmin, xmax, ymin, ymax) and centroid (x, y)
        self.bounds = self.get_bounds(bounds=True)
        self.centroid = (float(self.bounds[0] + self.bounds[1]) / 2.0,
                         float(self.bounds[2] + self.bounds[3]) / 2.0)

        # sizes of Tile along x and y in spatial reference units
        self.sizex = self.bounds[1] - self.bounds[0]
        self.sizey = self.bounds[3] - self.bounds[2]


class TileGrid(object):

    """
    Class to store and manipulate 2D grid of Tile objects.
    This class assumes that all Tile objects are
    in the same spatial reference system and have
    the same dimensions.
    """

    def __init__(self,
                 tiles=None):

        self.tiles = tiles  # list of Tile objects
        self.ntiles = len(tiles)  # number of tiles

        self.tile_bounds = None  # list of tile bounds
        self.tile_centroids = None  # list of tile centroids

        self.grid = None  # list of list of Tile objects
        self.grid_index = None  # number of tiles in each grid "cell"

        self.grid_extent = None  # xmin, xmax, ymin, ymax spatial extent of the grid
        self.grid_sizex = None  # Number of Tile "cells" along x
        self.grid_sizey = None  # Number of Tile "cells" along y

    def __add__(self, other):
        """
        Method to add one TileGrid to another
        """
        # add all tile layers
        # compute edges again for the output

        pass

    def __sub__(self, other):
        """
        Method to subtract one TileGrid from another
        """
        # subtract all layers from corresponding layers
        # re compute edges

        pass

    def make_void_layers(self, other):
        """
        Method to prepare a TileGrid layer of voids only
        """
        # make a duplicate TileGrid
        # read all tiles and find voids
        # store only the voids in other layer
        pass

    def apply_void_layers(self, other):
        """
        Method to apply the TileGrid layer of voids to self or another TileGrid object
        """
        # make void layer
        # mask the void layer in current TileGrid

        pass

    def get_tile_bounds(self):
        """
        Method to assign bounds and centroids of all Tiles in the list
        to the properties tile_bounds and tile_centroids respectively
        :return:
        """
        if (self.tiles is not None) and (len(self.tiles) > 0):

            # ntiles x 4 array of [xmin, xmax, ymin, ymax]
            self.tile_bounds = np.array(list(tile.bounds for tile in self.tiles))

            # ntiles x 2 array of [centroidX, centroidY]
            self.tile_centroids = np.array(list(tile.centroid for tile in self.tiles))

        else:
            raise ValueError("Tile list is empty")

    def get_extent(self):
        """
        Method to get spatial extent of TileGrid object. This method assumes
        that all Tiles are in the same spatial reference system.
        :return: [xmin, xmax, ymin, ymax]
        """
        self.get_tile_bounds()

        # get extent for the grid using extent of all tiles
        self.grid_extent = [np.min(self.tile_bounds[:, 0]),
                            np.max(self.tile_bounds[:, 1]),
                            np.min(self.tile_bounds[:, 2]),
                            np.max(self.tile_bounds[:, 3])]

        # size of grid in spatial reference units
        self.grid_sizex = round(float(self.grid_extent[1] - self.grid_extent[0]) / float(self.tiles[0].sizex))
        self.grid_sizey = round(float(self.grid_extent[3] - self.grid_extent[2]) / float(self.tiles[0].sizey))

    def make_grid(self):

        """
        Method to make grids (list of lists) and populate them with Tile objects
        retaining the spatial arrangement of the Tile objects
        :return: None
        """

        self.get_extent()

        # make empty list of lists for the grid
        self.grid = list(list(None for _ in range(self.grid_sizex)) for _ in range(self.grid_sizey))

        # index the number of tiles at each location
        self.grid_index = list(list(0 for _ in range(self.grid_sizex)) for _ in range(self.grid_sizey))

        # get x coords of grid lines
        end_y_coords = [0, self.grid_sizey]
        interp_func_y = interp1d(end_y_coords, [self.grid_extent[3], self.grid_extent[2]])
        mid_y_coords = interp_func_y(list(range(1, self.grid_sizey)))
        y_coord_list = [self.grid_extent[2]] + mid_y_coords + [self.grid_extent[3]]

        # get y coords of grid lines
        end_x_coords = [0, self.grid_sizex]
        interp_func_x = interp1d(end_x_coords, [self.grid_extent[1], self.grid_extent[0]])
        mid_x_coords = interp_func_x(list(range(1, self.grid_sizex)))
        x_coord_list = [self.grid_extent[0]] + mid_x_coords + [self.grid_extent[1]]

        # populate tile grids
        for tile_indx, centroid in enumerate(self.grid_extent.tolist()):
            for grid_x_indx in range(self.grid_sizex):
                if x_coord_list[grid_x_indx] < centroid[0] < x_coord_list[grid_x_indx + 1]:
                    for grid_y_indx in range(self.grid_sizey):
                        if y_coord_list[grid_y_indx] < centroid[1] < y_coord_list[grid_y_indx + 1]:
                            self.grid[grid_y_indx][grid_x_indx] = self.tiles[tile_indx]
                            self.grid_index[grid_y_indx][grid_x_indx] += 1

    @staticmethod
    def fill_adjacent_edges(tile,
                            next_tile,
                            edge_axis=0):

        """
        Method to compare and fill edge discontinuities (voids) that share the
        edges of two adjacent tile objects
        :param tile: Tile object
        :param next_tile: Tile object adjacent to first tile object, touching one edge
        :param edge_axis: Axis of edge alignment (default: 0)
                          0 = x-axis (l or r edge),
                          1 = y-axis (t or b edge)

        :returns: tuple of tile objects
        """

        if edge_axis not in (0, 1):
            raise ValueError("Invalid edge axis")

        edge_list = [('r', 'l'), ('b', 't')]

        size_list = [tile.metadata['ncols'], tile.metadata['nrows']]

        edges = edge_list[edge_axis]
        arr_size = size_list[edge_axis]

        tile_edge, tile_edge_loc = tile.edges[edges[0]]
        next_tile_edge, next_tile_edge_loc = next_tile.edges[edges[1]]

        for i in range(len(tile_edge_loc)):
            if not np.isnan(tile_edge_loc[i]) and not np.isnan(next_tile_edge_loc[i]):
                f = interp1d([tile_edge_loc[i] - arr_size, next_tile_edge_loc[i]],
                             [tile_edge[i], next_tile_edge[i]])
                tile_edge[i], next_tile_edge[i] = f(-1), f(0)
                tile_edge_loc[i], next_tile_edge_loc[i] = arr_size, 0

        tile.edges[edges[0]] = tile_edge, tile_edge_loc
        next_tile.edges[edges[1]] = next_tile_edge, next_tile_edge_loc

        return tile, next_tile

    def fill_multi_tile_voids(self,
                              axis=0):
        """
        Method to fill voids across more than two tiles
        :param axis: Axis along which the voids are to be filled
        """
        for line_indx in self.grid_sizey:
            tiles = self.grid[line_indx]

        if axis == 0:
            edge_keys = ('l', 'r')
        else:
            edge_keys = ('t', 'b')

        # look at all til edges in a row pixel by pixel
        # find all the tiles that are adjacent/consecutive
        # group adjacent tiles together and you have range of discontinuity
        # fill edges by interpolation
        # fill layers by interpolation
        # repeat for other direction
        # take mean of both directions



