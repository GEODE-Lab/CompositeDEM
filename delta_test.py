from dem_code.delta import FillVoids, np


if __name__ == '__main__':

    ar = np.array([[8,  7,  9,  3,  3, 4,  5],
                   [9,  6,  4, -1,  8, 9,  9],
                   [2, -1,  5, -1, -1, 4,  9],
                   [2, -1,  0,  3, -1, -1, 4],
                   [4, -1, -1,  8,  2,  4, 8],
                   [9,  3,  8,  2,  4,  6, 8]])

    print(ar)

    print('\n')

    arr1 = FillVoids(array=ar, nodata=-1)

    arr1x = np.apply_along_axis(arr1.fill_voids_1d,
                                1,
                                arr1.array,
                                arr1.nodata)

    print(arr1x)
    print('\n')

    arr1y = np.apply_along_axis(arr1.fill_voids_1d,
                                0,
                                arr1.array,
                                arr1.nodata)

    print(arr1y)
    print('\n')

    arr1f = arr1.fill()
    print(arr1f)
