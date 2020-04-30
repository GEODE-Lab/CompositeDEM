from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from demLib.common import Common, File
from demLib.spatial import Raster
from demLib.exceptions import AxisError, FileNotFound, FieldError, ProcessingError
import warnings
import numpy as np
import json


__all__ = ['Layer',
           'Tile',
           'Edge',
           'TileGrid']


class Layer(object):

    """
    Class to store and manipulate Raster Tile as a numpy array layer
    """

    def __init__(self,
                 array=None,
                 nodata=None):
        """
        Instantiate class
        :param array: numpy array (1D)
        :param nodata:
        """

        self.array = array
        self.ncol = None
        self.nrows = None
        self.nodata = nodata

        if self.array is not None:
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
        :param arr: Input 1-D numpy array
        :param nodata: no data value
        :return: List of tuples [(pixel loc before void, pixel loc after void), ]
        """

        void_locs = np.msort(np.where(arr == nodata)[0])
        grouped_locs = Common.group_consecutive(void_locs.tolist())

        blocks = []
        for group in grouped_locs:
            if len(group) > 0:
                blocks.append([group[0] - 1 if group[0] != 0 else nodata,
                               group[-1] + 1 if group[-1] != (arr.shape[0] - 1) else nodata])

        return blocks

    @staticmethod
    def fill_voids(arr,
                   nodata=None,
                   blocks=None,
                   loc_array=None):
        """
        Method to fill a void in a 1-D array
        if none of the block indices are negative
        :param arr: 1-D numpy array
        :param nodata: no data value
        :param blocks: List of tuples of block locations, output from find_blocks()
        :param loc_array: Location of pixels
        :return: 1-D array
        """
        out_arr = arr.copy()

        if blocks is not None and len(blocks) > 0:

            for block in blocks:

                if not any([(indx == nodata) for indx in block]):
                    if loc_array is not None:
                        arr_block = loc_array[block]
                    else:
                        arr_block = block

                    y = out_arr[list(block)]
                    f = interp1d(arr_block, y)

                    if loc_array is not None:
                        out_arr[np.arange(*block)] = f(loc_array[np.arange(*block)])
                    else:
                        out_arr[np.arange(*block)] = f(np.arange(*block))
                else:
                    if block[0] != nodata:
                        out_arr[block[0]: -1] = out_arr[block[0]]
                    elif block[1] != nodata:
                        out_arr[0: block[1]] = out_arr[block[1]]

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
        blocks = Layer.find_blocks(arr,
                                   nodata)

        return Layer.fill_voids(arr,
                                nodata=nodata,
                                blocks=blocks)

    @staticmethod
    def fill_voids_by_loc(arr,
                          nodata,
                          loc_array=None):
        """
        Method to fill all voids in a 1D array
        :param arr: 1-D array
        :param nodata: No data value
        :param loc_array: Location of pixels
        :return: 1-D array
        """
        return Layer.fill_voids(arr,
                                nodata=nodata,
                                blocks=Layer.find_blocks(arr,
                                                         nodata),
                                loc_array=loc_array)

    def fill(self,
             bidirectional=True,
             smooth=False,
             sigma=3):
        """
        Method to fill voids in 2D array by
            Filling voids in 1D array along x axis
            Filling voids in 1D array along y axis
            and taking the mean of two 2D arrays
            This methods leaves out voids at edges
        :return: 2D array
        """
        if len(self.array.shape) == 3:
            x_axis, y_axis = 1, 2
        else:
            x_axis, y_axis = 0, 1

        xfilled_arr = np.apply_along_axis(self.fill_voids_1d,
                                          x_axis,
                                          self.array,
                                          self.nodata)

        yfilled_arr = np.apply_along_axis(self.fill_voids_1d,
                                          y_axis,
                                          self.array,
                                          self.nodata)

        x_remain_voids = np.where(xfilled_arr == self.nodata)
        y_remain_voids = np.where(yfilled_arr == self.nodata)

        mean_arr = (xfilled_arr + yfilled_arr)/2.0

        if smooth:
            smoothed_mean_arr = gaussian_filter(mean_arr,
                                                sigma=sigma)
            void_loc = np.where(self.array == self.nodata)
            mean_arr[void_loc] = smoothed_mean_arr[void_loc]

        if bidirectional:
            mean_arr[x_remain_voids] = self.nodata
            mean_arr[y_remain_voids] = self.nodata

        self.array = mean_arr


class Edge(object):
    """
    Class for storing and processing edge values for a Raster Tile
    """
    def __init__(self,
                 edges=None,
                 filename=None,
                 nodata=None):
        """
        Instantiate Edge class
        :param edges: Dictionary to store edge information in the following format:
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

        self.edges = edges
        self.filename = filename
        self.nodata = nodata

    def __repr__(self):
        return '<Edge object {}>'.format('-MEM-' if self.filename is None else self.filename)

    def load_edges(self,
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
                raise FileNotFound('No valid file specified')

        with open(self.filename, 'r') as ef:
            self.edges = json.loads(ef.read())

        self.nodata = self.edges['n']

    def write_edges(self,
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
                raise FileNotFound('No valid file specified')

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
        arr_end_locs = [0, arr.shape[0]-1]
        blocks = Layer.find_blocks(arr,
                                   nodata)
        if len(blocks) == 0:
            return np.hstack([arr[arr_end_locs],
                              np.array(arr_end_locs)])

        else:
            first_block_end = blocks[0][0]
            last_block_end = blocks[-1][1]

            if first_block_end >= arr_end_locs[0]:
                arr_first_loc = arr_end_locs[0]
                if last_block_end <= arr_end_locs[1]:
                    arr_last_loc = arr_end_locs[1]
                else:
                    arr_last_loc = first_block_end
            else:
                if last_block_end <= arr_end_locs[1]:
                    arr_first_loc = last_block_end
                    arr_last_loc = arr_end_locs[1]
                else:
                    arr_first_loc = arr_end_locs[0]
                    arr_last_loc = arr_end_locs[1]

            return np.array([nodata if arr_first_loc == nodata else arr[arr_first_loc],
                             nodata if arr_last_loc == nodata else arr[arr_last_loc],
                             arr_first_loc,
                             arr_last_loc])

    def extract_edges(self,
                      array=None):
        """
        Method to extract edges from a tile
        :param array: 2D array
        :return: edge dict
        """

        if array is None:
            if hasattr(self, 'array'):
                array = self.array
            else:
                raise ProcessingError("No array to extract edges from.")

        # value and location array for left and right edges
        val_loc_arr_lr = np.apply_along_axis(lambda x: self.get_nearest_loc_val(x, self.nodata), 1, array)

        # value and location array for top and bottom edges
        val_loc_arr_tb = np.apply_along_axis(lambda x: self.get_nearest_loc_val(x, self.nodata), 1, array.T)

        edge_dict = dict()

        edge_dict['l'] = [val_loc_arr_lr[:, 0].tolist(), val_loc_arr_lr[:, 2].tolist()]
        edge_dict['r'] = [val_loc_arr_lr[:, 1].tolist(), val_loc_arr_lr[:, 3].tolist()]

        edge_dict['t'] = [val_loc_arr_tb[:, 0].tolist(), val_loc_arr_tb[:, 2].tolist()]
        edge_dict['b'] = [val_loc_arr_tb[:, 1].tolist(), val_loc_arr_tb[:, 3].tolist()]

        edge_dict['n'] = self.nodata

        self.edges = edge_dict


class Tile(Raster, Edge, Layer):

    """
    Class to amalgamate properties and methods of
    Raster, Edge, and Layer classes into a Tile object
    """

    def __init__(self,
                 filename=None,
                 edgefile=None,
                 nodata=None,
                 get_array=True):
        """
        Instantiate Tile class
        :param filename: Tile filepath (usually a .tif file)
        :param edgefile: filepath of .edge file
        :param nodata: No data value to use for voids
        """
        self.array = None

        Raster.__init__(self,
                        filename=filename,
                        get_array=get_array)

        if nodata is not None:
            self.nodata = nodata
        elif self.metadata['nodatavalue'] is not None:
            self.nodata = self.metadata['nodatavalue']
        else:
            self.nodata = None
            warnings.warn("No-data value is not defined",
                          stacklevel=2)

        Edge.__init__(self,
                      filename=edgefile,
                      nodata=self.nodata)
        Layer.__init__(self,
                       array=self.array,
                       nodata=self.nodata)

        if edgefile is not None and File(edgefile).file_exists():
            Edge.load_edges(self,
                            filename=edgefile)

        elif self.filename is not None:
            if get_array:
                Layer.__init__(self,
                               array=self.array,
                               nodata=self.nodata)

        # geometric properties of the Tile object:
        # bounds: (xmin, xmax, ymin, ymax) and centroid (x, y)
        if self.metadata['transform'] is not None:
            self.bounds = self.get_bounds(bounds=True)
            self.centroid = (float(self.bounds[0] + self.bounds[1]) / 2.0,
                             float(self.bounds[2] + self.bounds[3]) / 2.0)

            # sizes of Tile along x and y in spatial reference units
            self.sizex = self.bounds[1] - self.bounds[0]
            self.sizey = self.bounds[3] - self.bounds[2]

        if self.array is not None and self.nodata is not None:
            self.void_loc = np.where(self.array == self.nodata)
        else:
            self.void_loc = None

    def __repr__(self):
        """
        String representation of the Tile object
        """
        filename = getattr(self, 'filename', None)
        edgefile = getattr(self, 'edgefile', None)
        array = getattr(self, 'array', None)
        nodata = getattr(self, 'nodata', None)

        if array is not None:
            shape = ' x '.join([str(elem) for elem in array.shape])
        else:
            shape = None

        return "<Tile object with shape: {}, nodata: {}, filename: {}, edgefile: {} >".format(shape,
                                                                                              nodata,
                                                                                              filename,
                                                                                              edgefile)

    def resample(self,
                 other,
                 alg='near'):
        """
        Re-sample tile other to match tile of self.

        :param other: The other tile to match the spatial resolution with
        :param alg: Interpolation algorithm : { near, bilinear, cubic (default) }

        :returns: numpy array
        """
        if self.metadata['ncols'] != other.metadata['ncols'] or \
                self.metadata['nrows'] != other.metadata['nrows']:
            # other = self.resample(other)
            alg_options = {'near': 0, 'bilinear': 1, 'cubic': 2}

            zoom_factor = [1,
                           float(self.metadata['nrows'])/float(other.metadata['nrows']),
                           float(self.metadata['ncols'])/float(other.metadata['ncols'])]

            # output_size = self.metadata['nrows'], self.metadata['ncols']

            out_shape = (np.array(other.array.shape) * np.array(zoom_factor)).astype(np.int16)

            # other.array[np.where(other.array == other.nodata)] = np.NaN
            # print(other.array.shape)
            # print(out_shape)
            '''
            other.array = zoom(input=other.array,
                               zoom=zoom_factor,
                               order=alg_options[alg],
                               mode='nearest',
                               prefilter=True)
            '''
            other.array = resize(other.array,
                                 output_shape=out_shape,
                                 order=alg_options[alg])

            # print(other.array.shape)

            # other.array[np.where(other.array == np.NaN)] = other.nodata

            other.metadata['nbands'], other.metadata['nrows'], other.metadata['ncols'] = \
                other.array.shape

            other.metadata['transform'] = self.metadata['transform']

            other.void_loc = np.where(other.array == other.nodata)

        return other

    def __add__(self,
                second):
        """
        Method to add one Tile to another
        :param second: second tile or number in the equation
        :returns: Tile object
        """

        result = Tile()

        result.__dict__.update(self.__dict__)

        if isinstance(second, Tile):
            if self.metadata['ncols'] != second.metadata['ncols'] or \
                    self.metadata['nrows'] != second.metadata['nrows']:
                # other = self.resample(other)
                raise ProcessingError('Unequal tile sizes')
            else:
                result.array = self.array + second.array

        elif type(second) in (float, int):
            result.array = self.array + second
        else:
            raise ProcessingError("Unsupported data type for add")

        # if self.void_loc is not None:
        #    result.array[self.void_loc] = self.nodata

        # if other.void_loc is not None:
        #    result.array[self.void_loc] = self.nodata

        return result

    def __sub__(self,
                second):
        """
        Method to subtract one Tile from another
        :param second: second tile or number in the equation
        :returns: Tile object
        """

        result = Tile()

        result.__dict__.update(self.__dict__)

        if isinstance(second, Tile):
            if self.metadata['ncols'] != second.metadata['ncols'] or \
                    self.metadata['nrows'] != second.metadata['nrows']:
                raise ProcessingError('Unequal tile sizes')
            else:
                # other = self.resample(other)
                # other.write_raster('D:/temp/dem_tiles/other_after_zoom5.tif')
                # exit()
                result.array = self.array - second.array

        elif type(second) in (float, int):
            result.array = self.array - second
        else:
            raise ProcessingError("Unsupported data type for subtract")

        # if self.void_loc is not None:
        #    result.array[self.void_loc] = self.nodata

        # if other.void_loc is not None:
        #    result.array[self.void_loc] = self.nodata

        return result

    @classmethod
    def void_tile(cls,
                  first,
                  second):
        """
        Method to subtract one Tile from another
        :param first: first tile in the equation
        :param second: second tile or number in the equation
        :returns: Tile object
        """
        tile = cls()
        tile.__dict__.update(first.__dict__)
        tile.array = np.zeros(first.array.shape, dtype=np.uint8)
        tile.array[np.where(first.array == first.nodata)] += 1
        tile.array[np.where(second.array == second.nodata)] += 1

        tile.void_loc = np.where(tile.array == 2)

        return tile

    def copy_voids(self,
                   other):
        """
        Method to copy voids in one Tile to another
        :param other: Other tile object
        :returns: None
        """

        if isinstance(other, Tile):
            if self.metadata['ncols'] != other.metadata['ncols'] or \
                    self.metadata['nrows'] != other.metadata['nrows']:
                raise ProcessingError("Unequal tile sizes for copy")
            else:
                self.array[other.void_loc] = self.nodata
        else:
            raise ProcessingError("Unsupported data type for copy")

    def update_array(self,
                     edgefile=None):
        """
        Method to update tile array using edges (edge_dict)
        """
        if edgefile is not None and File(edgefile).file_exists():
            self.load_edges(edgefile)

        if self.edges is None:
            warnings.warn("Edge dictionary or file is missing in input. No edges loaded.",
                          stacklevel=2)
            return

        else:
            for key, edge_list in self.edges.items():
                val_arr = np.array(edge_list[0])
                loc_arr = np.array(edge_list[1])

                if key == 'l':
                    nodata_loc = np.where(loc_arr != 0)
                    val_arr[nodata_loc] = self.nodata
                    self.array[:, 0] = val_arr

                elif key == 'r':
                    nodata_loc = np.where(loc_arr != self.metadata['ncols'])
                    val_arr[nodata_loc] = self.nodata
                    self.array[:, self.metadata['ncols']] = val_arr

                elif key == 't':
                    nodata_loc = np.where(loc_arr != 0)
                    val_arr[nodata_loc] = self.nodata
                    self.array[0, :] = val_arr

                elif key == 'b':
                    nodata_loc = np.where(loc_arr != self.metadata['nrows'])
                    val_arr[nodata_loc] = self.nodata
                    self.array[self.metadata['nrows'], :] = val_arr

                else:
                    raise KeyError("Invalid keys in edge dictionary")


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
        self.grid_sizex = None  # Number of Tiles along x
        self.grid_sizey = None  # Number of Tiles along y

        self.nodata = None

        self.adjacent_filled = False
        self.multi_filled = True

    def __repr__(self):
        return self.__class__.__name__

    def get_tile_bounds(self):
        """
        Method to assign bounds and centroids of all Tiles in the list
        to the properties tile_bounds and tile_centroids respectively
        :return:
        """
        if (self.tiles is not None) and (len(self.tiles) > 0):

            # ntiles x 4; array of [xmin, xmax, ymin, ymax]
            self.tile_bounds = np.array(list(tile.bounds for tile in self.tiles))

            # ntiles x 2 array of [centroidX, centroidY]
            self.tile_centroids = np.array(list(tile.centroid for tile in self.tiles))

        else:
            raise FieldError("Tile list is empty")

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
        self.grid = np.array(list(list(None for _ in range(self.grid_sizex)) for _ in range(self.grid_sizey)))

        # index the number of tiles at each location
        self.grid_index = np.array(list(list(0 for _ in range(self.grid_sizex)) for _ in range(self.grid_sizey)))

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
                            self.grid[grid_y_indx, grid_x_indx] = self.tiles[tile_indx]
                            self.grid_index[grid_y_indx, grid_x_indx] += 1

    def get_next_tile(self,
                      axis=1,
                      separator=True,
                      separator_value=0):
        """
        Method to return next tile in a grid with separator at each new line (row)
        :param axis: Axis of operation (0=along rows, 1=along columns)
        :param separator: If the new line (row) should have a separator
        :param separator_value: value of the new line (row) separator
        """
        if axis == 0:
            grid = self.grid.T
        elif axis == 1:
            grid = self.grid
        else:
            raise ValueError("Axis index must be 0 or 1")

        for row_indx in range(grid.shape[0]):
            for col_indx in range(grid.shape[1]):

                if separator and (col_indx == grid.shape[1]):
                    yield separator_value
                else:
                    yield grid[row_indx, col_indx]

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
            raise AxisError

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

    def fill_multi_tile_void_edges(self,
                                   axis=1):
        """
        Method to fill edges of tiles with voids across more than two tiles
        :param axis: Axis along which the voids are to be filled
        """
        if axis == 1:
            grid_rows = self.grid_sizey
            grid_cols = self.grid_sizex
            edge_keys = ['l', 'r']
            grid = self.grid

        elif axis == 0:
            grid_rows = self.grid_sizex
            grid_cols = self.grid_sizey
            edge_keys = ['t', 'b']
            grid = self.grid.T

        else:
            raise ValueError('Axis can only be 0 or 1')

        arr_ncols = 2 * grid_cols

        for row_indx in range(grid_rows):

            arr_nrows = grid[row_indx, 0].shape[0]

            edge_list = []
            val_list = []
            for col_indx in range(grid_cols):
                tile = grid[row_indx, col_indx]

                edge_list.append(tile.edges[edge_keys[0]][0])
                edge_list.append(tile.edges[edge_keys[1]][0])

                val_list.append(tile.edges[edge_keys[0]][0])
                val_list.append(tile.edges[edge_keys[1]][0])

            edge_arr = np.array(edge_list, dtype=np.float32).T
            val_arr = np.array(val_list, dtype=np.int32).T

            if edge_arr.shape[0] != arr_nrows or edge_arr.shape[1] != arr_ncols:
                raise ProcessingError("Incorrect tile information in grid")

            for row in range(edge_arr.shape[1]):
                temp_arr = Layer.fill_voids_by_loc(edge_arr[row, :],
                                                   self.nodata,
                                                   val_arr[row, :])
                edge_arr[row, :] = temp_arr

            edge_counter = 0
            for col_indx in range(grid_cols):
                grid[row_indx, col_indx].edges[edge_keys[0]] = edge_arr[:, edge_counter]
                grid[row_indx, col_indx].edges[edge_keys[1]] = edge_arr[:, edge_counter + 1]
                edge_counter += 2

    def fill_adjacent(self):
        """
        Method to fill voids on adjacent edges in a TileGrid object
        """

        if self.adjacent_filled:
            warnings.warn("Adjacent tiles are already filled.\n" +
                          "Set flag {}.adjacent_filled=False for re-filling".format(self))
            return

        self.adjacent_filled = True

        if self.grid is None:
            self.make_grid()

        for axis in [0, 1]:
            tile = self.get_next_tile(axis)
            for nxt_tile in self.get_next_tile(axis):
                if isinstance(nxt_tile, Tile):
                    self.fill_adjacent_edges(tile, nxt_tile, axis)
                    tile = nxt_tile
                else:
                    tile = self.get_next_tile(axis)

    def fill_multi(self):
        """
        Method to fill multi tile voids in a TileGrid object
        """

        if self.multi_filled:
            warnings.warn("Multiple tile voids are already filled.\n"
                          "Set flag {}.multi_filled=False for re-filling".format(self))

        elif not self.adjacent_filled:
            warnings.warn("Adjacent tile voids unfilled. Filling them first\n")
            self.fill_adjacent()

        else:
            self.multi_filled = True

            if self.grid is None:
                self.make_grid()

            for axis in [0, 1]:
                self.fill_multi_tile_void_edges(axis)

