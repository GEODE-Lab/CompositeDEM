from scipy.interpolate import interp1d
from geosoup import Raster
import numpy as np
import json


class Layer(object):

    def __init__(self,
                 array=None,
                 edgex=None,
                 edgey=None,
                 nodata=None):
        """
        Class to hold numpy array, its edges and no data object
        :param array: numpy array (1D)
        :param edgey: The two edges of the array (nrows x 2) columns:
                        1) left
                        2) right

        :param edgex: The two edges of the array (2 x ncol) rows:
                       1) top
                       2) bottom

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
        loc = np.msort(np.where(arr == nodata)[0])

        if loc.shape[0] > 0:
            arr_mask = arr.copy()
            arr_mask = arr_mask * 0 + 1
            arr_mask[loc] = 0

            first_jump = -1 if arr_mask[0] == 0 else 0

            arr_jumps = np.hstack([first_jump, arr_mask[1:] - arr_mask[:-1]])

            jump_starts = (np.where(arr_jumps == -1)[0] - 1).tolist()
            jump_ends = (np.where(arr_jumps == 1)[0]).tolist()

            if len(jump_starts) != len(jump_ends):
                jump_ends.append(-1)

            return list(zip(jump_starts, jump_ends))

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

        print(blocks)

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
    Class for storing and processing edge values for a tile
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
                Here, all the edge_vals and the edge_locs are 1D list objects
                all elements should be present. Absence of even one element makes the
                dictionary incomplete and invalid. Unavailability of any element should be
                indicated by -1.

        :param filename: filename for edge_dict
        :param nodata: No data value
        """

        self.edges = edge_dict
        self.filename = filename
        self.nodata = nodata

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
        function to get nearest located non nodata value and its location
        for two ends of a 1D array
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
    Class to amalgamate properties and methods of Raster, Edge, and Layer classes
    """

    def __init__(self,
                 name,
                 edgefile=None,
                 nodata=None):
        """
        Instantiate Tile class
        :param name: Name of the Tile or filepath
        :param edgefile: filepath of edges file
        :param nodata: No data value to use for voids
        """

        Raster.__init__(self,
                        name)
        self.initialize(get_array=True)

        Edge.__init__(self,
                      filename=edgefile,
                      nodata=nodata)

        if edgefile is not None:
            Edge.load(self,
                      filename=edgefile)
        else:
            Edge.extract(self,
                         self.array[0, :, :])

        Layer.__init__(self,
                       array=self.array,
                       nodata=nodata)


class TileGrid(Tile):

    def __init__(self):
        pass


