from demLib.tilegrid import Layer, np, Edge


if __name__ == '__main__':

    ar = np.array([[8,  7,  9,  3,  3, 4,  5],
                   [9,  6,  4, -1,  8, 9,  9],
                   [2, -1,  5, -1, -1, 4,  9],
                   [-1, -1,  -1,  -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, 11],
                   [4, -1, -1,  8,  2,  4, 8],
                   [9,  3,  8,  2,  4,  6, 8]])

    edge = Edge(nodata=-1)
    edge.extract(ar)

    print(edge.edges)

    print(edge)

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

