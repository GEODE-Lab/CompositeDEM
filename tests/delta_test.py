from demLib.tilegrid import Layer, np, Edge


if __name__ == '__main__':

    ar = np.array([[8,   7,   9,   3,   4,   3,   4,   5],
                   [9,   6,   4,  -1,  -1,   8,   9,   9],
                   [2,  -1,  -1,  -1,  -1,  -1,   4,   9],
                   [-1, -1,   1,   1,  -1,   1,  -1,  -1],
                   [-1, -1,  -1,  -1,  -1,  -1,  -1,  11],
                   [4,  -1,  -1,   8,   4,   2,   4,   8],
                   [9,   3,   8,   2,   5,   4,   6,   8]])

    cl = np.array([[0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80],
                   [0,  20,  21,  40,  41,  60,  61,  80]])

    nodata = -1

    edge = Edge(nodata=nodata)
    edge.extract_edges(ar)

    print(edge)
    print(edge.edges)

    layer = Layer(array=ar,
                  nodata=nodata)

    filled_layer = layer.fill()

    print(layer)
    print(layer.array)
    print(filled_layer)

    combined_arr = np.array([ar, cl])
    print(combined_arr)
    print(combined_arr.shape)

    out_arr = ar.copy() * 0.0

    for row in range(combined_arr.shape[1]):
        print(combined_arr[0, row, :])
        print(combined_arr[1, row, :])
        print('-------')
        temp_arr = Layer.fill_voids_by_loc(combined_arr[0, row, :],
                                           nodata,
                                           combined_arr[1, row, :])
        out_arr[row, :] = temp_arr

    print(out_arr)


