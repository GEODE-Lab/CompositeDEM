from scipy.interpolate import interp1d
import numpy as np


class FillVoids(object):

    def __init__(self,
                 array=None,
                 nodata=None):
        self.array = array
        self.ncol = array.shape[1]
        self.nrows = array.shape[0]
        self.nodata = nodata

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

            arr_jumps = np.hstack([0, arr_mask[1:]-arr_mask[:-1]])

            jump_starts = (np.where(arr_jumps == -1)[0] - 1).tolist()
            jump_ends = (np.where(arr_jumps == 1)[0]).tolist()

            if len(jump_starts) != len(jump_ends):
                jump_ends.append(arr.shape[0] - 1)

            if jump_starts[0] == -1:
                jump_starts[0] = 0

            return zip(jump_starts, jump_ends)

    @staticmethod
    def fill_voids(arr,
                   blocks=None):
        """
        Method to fill voids in a 1-D array
        :param arr: 1-D array
        :param blocks: List of tuples of block locations, output from find_blocks()
        :return: 1-D array
        """
        out_arr = arr.copy()

        if blocks is not None and len(blocks) > 0:
            for block in blocks:
                y = out_arr[list(block)]
                f = interp1d(block, y)
                out_arr[np.arange(*block)] = f(np.arange(*block))

        return out_arr

    @staticmethod
    def fill_voids_1d(arr,
                      nodata):
        """
        Method to fill voids in 1D array by
                filling voids in 1d arrays
        :param arr: 1-D array
        :param nodata: No data value
        :return: 1-D array
        """
        void_blocks = FillVoids.find_blocks(arr,
                                            nodata)
        return FillVoids.fill_voids(arr,
                                    void_blocks)

    def fill(self):
        """
        Method to fill voids in 2D array by
            Filling voids in 1D array along x axis
            Filling voids in 1D array along y axis
            and taking the mean of two 2D arrays
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

        return (xfilled_arr + yfilled_arr)/2.0


class FillEdges(object):

    def __init__(self):
        pass



