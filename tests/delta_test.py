from demLib.tilegrid import Layer, np, Edge


if __name__ == '__main__':

    ar = np.array([[8,   7,   9,   3,   4,   3,   4,   5],
                   [9,   6,   4,  -1,   0,   8,   9,   9],
                   [2,  -1,   5,  -1,  -1,  -1,   4,   9],
                   [-1, -1,  -1,  -1,  -1,  -1,  -1,  -1],
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
    edge.extract(ar)

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





    exit()


    print(ar)

    print('\n')

    arr1 = Layer(array=ar, nodata=-1)

    print(arr1)
    print(Layer.find_blocks(ar[2,:], -1))


    arr1f = arr1.fill()
    print(arr1f)
    print('------------------')

    lw = np.array([
        Edge.get_nearest_loc_val(ar[0, :], -1),
        Edge.get_nearest_loc_val(ar[1, :], -1),
        Edge.get_nearest_loc_val(ar[2, :], -1),
        Edge.get_nearest_loc_val(ar[3, :], -1),
        Edge.get_nearest_loc_val(ar[4, :], -1),
        Edge.get_nearest_loc_val(ar[5, :], -1),
        Edge.get_nearest_loc_val(ar[6, :], -1),

    ])

    print(lw)

    exit()


    a=[[1,2],[2,4],[1,9],[1,5],[2,1],[1,4],[3,2],[3,1],[2,2],[2,3]]

    b = sorted(a, key=lambda x: (x[0], x[1]))

    print(b)

