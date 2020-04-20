from osgeo import osr, ogr
import multiprocessing as mp
from sys import argv
import pickle
import json
import sys
import time
import gc
from demLib.common import File, group_multi, Timer
from demLib.spatial import Vector
from data import above_coords
from data import can_file_dir, tile_file, out_file, out_dir
# pre_merge_border_vec, post_merge_border_vec, \
# ak_file


# buffer distance for geometry touching/intersecting check
buffer_dist = 1  # meters


def multi_feat_union(list_of_geom_indx_lists):
    """
    Function to loop through all the features that intersect/touch and dissolve them
    :param list_of_geom_indx_lists: List of list of geometry indices
    :returns list of dicts
    """

    dissolved_features = list()
    for geom_indx_list in list_of_geom_indx_lists:

        # dissolved feature is named after
        # the feature with the largest area
        features_to_dissolve = sorted(list(feat_dict[indx] for indx in geom_indx_list),
                                      key=lambda k: k['area'],
                                      reverse=True)

        # use id and name of the first feature
        feat_id = features_to_dissolve[0]['orig_id']
        feat_filename = features_to_dissolve[0]['filename']

        # create empty multi geometry
        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        # add features
        tiles = list()
        for feature in features_to_dissolve:
            tmp_geom = ogr.CreateGeometryFromWkt(feature['geom'])
            multi_geom.AddGeometryDirectly(tmp_geom.Buffer(buffer_dist))
            tiles = tiles + feature['tiles']

        # list of tiles that the dissolved feature intersects
        # is the union of the intersectino tile lists of all constituent features
        tiles = list(set(tiles))

        # take geom union and calc area-
        grp_geom = multi_geom.UnionCascaded()
        grp_area = grp_geom.GetArea()

        # add to list
        dissolved_features.append({'geom': grp_geom.ExportToWkt(),
                                   'filename': feat_filename,
                                   'orig_id': feat_id,
                                   'area': grp_area,
                                   'tiles': tiles})

    return dissolved_features


def find_tile(args):

    """
    Method to extract the tile number of each of the lakes
    :param args: Arguments
        (fid, feat_wkt, tile_wkts, tile_attrs)
    :return: tuple
    """

    fead_id, feat_wkt, tile_wkts, tile_attributes = args

    temp_geom = ogr.CreateGeometryFromWkt(feat_wkt)

    list_of_tiles = list()
    for tile_indx, tile_wkt in enumerate(tile_wkts):
        if temp_geom.Intersects(ogr.CreateGeometryFromWkt(tile_wkt)):
            list_of_tiles.append(tile_attributes[tile_indx]['grid_id'])

    return fead_id, list_of_tiles


def find_intersecting(args):
    """
    Method to find intersecting lakes with a list of geometries
    :param args: Arguments
        (fid, geom_wkt, wktlist)
    :return: tuple
    """

    fead_id, geom_wkt, geom_wktlist = args

    check_geom = ogr.CreateGeometryFromWkt(geom_wkt).Buffer(buffer_dist)

    intersect_list = list()
    for indx, wkt in enumerate(geom_wktlist):
        temp_geom = ogr.CreateGeometryFromWkt(wkt)
        if check_geom.Intersects(temp_geom.Buffer(buffer_dist)):
            intersect_list.append(indx)

    return fead_id, intersect_list


if __name__ == '__main__':

    t = time.time()

    # size = mp.cpu_count()
    size = 4
    pool = mp.Pool(processes=size)

    if len(argv) > 1:
        script, ak_file, can_file_dir, tile_file, \
            pre_merge_border_vec, post_merge_border_vec, \
            out_file, out_dir = argv

    print('CPUs: {}'.format(str(size)))

    error_list = list()
    border_list = list()

    geog_spref = osr.SpatialReference()
    geog_spref.ImportFromProj4('+proj=longlat +datum=WGS84')

    filename_attr = ogr.FieldDefn('filename', ogr.OFTString)
    id_attr = ogr.FieldDefn('orig_id', ogr.OFTString)
    area_attr = ogr.FieldDefn('area', ogr.OFTReal)
    area_attr.SetPrecision(8)
    area_attr.SetWidth(32)

    print('----------------------------------------------------------------')
    print('processing tile vector.....')

    # read tiles in memory
    tile_vec = Vector(filename=tile_file,
                      verbose=False)

    main_crs_str = tile_vec.spref_str

    print(tile_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing ABoVE boundary vector.....')

    # ABoVE vector in memory
    above_bounds_str = json.dumps({"type": "Polygon",
                                   "coordinates": above_coords})

    above_geom = ogr.CreateGeometryFromJson(above_bounds_str)
    above_geom.CloseRings()
    above_geom = Vector.reproj_geom(above_geom,
                                    geog_spref.ExportToWkt(),
                                    main_crs_str)

    above_vec = Vector(in_memory=True,
                       spref_str=geog_spref.ExportToWkt(),
                       geom_type=Vector.ogr_geom_type('polygon'))

    above_vec.name = 'above_bounds'
    above_vec.add_feat(above_geom)

    print(above_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing ak vector.....')

    sys.stdout.flush()
    '''
    ak_vec = Vector(in_memory=True,
                    spref_str=main_crs_str,
                    geom_type=Vector.ogr_geom_type('polygon'))

    ak_vec.name = 'alaska_lakes'

    ak_vec.layer.CreateField(filename_attr)
    ak_vec.layer.CreateField(id_attr)
    ak_vec.fields = ak_vec.fields + [filename_attr, id_attr]
    ak_vec.spref_str = main_crs_str

    temp_vec = None
    geom_list = list()

    filename = File(ak_file).basename.split('.shp')[0]

    try:
        temp_vec = Vector(ak_file,
                          spref_str=main_crs_str,
                          geom_type=Vector.ogr_geom_type('polygon'))

    except (AttributeError, RuntimeError):
        error_list.append('Error reading {}'.format(ak_file))
        print('Error reading {}'.format(ak_file))

    if temp_vec:
        if temp_vec.bounds.Intersects(above_geom):
            print(temp_vec)

            for feat in temp_vec.features:
                geom = feat.GetGeometryRef()
                geom_dict = feat.items()

                geom_list.append((str(geom_dict['OBJECTID']),
                                  geom.ExportToWkt(),
                                  tile_vec.wktlist,
                                  tile_vec.attributes))

                ak_vec.add_feat(geom=geom,
                                attr={'filename': filename,
                                      'orig_id': geom_dict['OBJECTID']})

            results = pool.map(find_tile,
                               geom_list)

            ak_vec.data.update(dict(results))
        else:
            print('Vector is outside AOI')

        temp_vec = None

    sys.stdout.flush()
    
    print(ak_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    '''
    print('processing can vector.....')

    sys.stdout.flush()

    can_file_obj = File()
    can_file_obj.dirpath = can_file_dir
    can_file_list = can_file_obj.find_all('*.shp')

    can_vec = Vector(in_memory=True,
                     spref_str=main_crs_str,
                     geom_type=Vector.ogr_geom_type('polygon'))

    can_vec.name = 'canada_lakes'
    can_vec.layer.CreateField(filename_attr)
    can_vec.layer.CreateField(id_attr)
    can_vec.fields = can_vec.fields + [filename_attr, id_attr]
    can_vec.spref_str = main_crs_str

    # loop through each file in the list
    for i, can_file in enumerate(can_file_list):
        filename = File(can_file).basename.split('.shp')[0]
        geom_list = list()

        print('File {} of {}: {}'.format(str(i+1),
                                         str(len(can_file_list)),
                                         can_file))
        temp_vec = None

        try:
            # read each file as vector
            temp_vec = Vector(filename=can_file,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))

        except (AttributeError, RuntimeError):
            error_list.append('Error reading {}'.format(can_file))
            print('Error reading {}'.format(can_file))

        if temp_vec:
            if temp_vec.bounds.Intersects(above_geom):
                print(temp_vec)

                # loop through each feature in the vector
                geom_list = list()
                for feat in temp_vec.features:
                    geom = feat.GetGeometryRef()
                    geom_dict = feat.items()

                    geom_list.append((str(geom_dict['NID']),
                                      geom.ExportToWkt(),
                                      tile_vec.wktlist,
                                      tile_vec.attributes))

                    can_vec.add_feat(geom=geom,
                                     attr={'filename': filename,
                                           'orig_id': geom_dict['NID']})

                # find the intersecting tiles for each feature
                for result in pool.map(find_tile, geom_list):
                    can_vec.data.update([result])
            else:
                print('Vector is outside AOI')

            temp_vec = None

        sys.stdout.flush()

    print(can_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

    for err in error_list:
        print(err)
    '''
    print('----------------------------------------------------------------')
    print('Spatial ref strings: \n')
    print(ak_vec.spref_str)
    print(can_vec.spref_str)
    print(tile_vec.spref_str)
    print('----------------------------------------------------------------')
    print('Merge Canada and Alaska lakes...')
    sys.stdout.flush()

    can_vec.merge(ak_vec, True)
    '''
    can_vec.name = 'ak_can_merged'
    print('Merged initial vector: {}'.format(can_vec))

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    all_tile_names = list(attr['grid_id'] for attr in tile_vec.attributes)
    print('Get all tile names...completed at {}'.format(Timer.display_time(time.time() - t)))
    sys.stdout.flush()

    # make a dictionary of all tiles as keys, and with values as dictionary of geometries
    tile_dict = dict()
    for key, vals in can_vec.data.items():
        for val in vals:
            if val not in tile_dict:
                tile_dict[val] = dict()
            tile_dict[val].update({key: dict()})
    print('Dictionary structure to store lakes by tile...completed at {}'.format(Timer.display_time(time.time() - t)))
    sys.stdout.flush()

    # dictionary of all features
    feat_dict = dict()
    feat_count = 0
    for feat in can_vec.features:
        geom = feat.GetGeometryRef()
        orig_id = feat.items()['orig_id']
        area = geom.GetArea()
        filename = feat.items()['filename']

        # get list of tile IDs that the geometry intersects
        tile_ids = can_vec.data[orig_id]

        if len(tile_ids) > 0:
            fid = feat_count
            print('ID: {} Area: {} Tiles: {}'.format(str(fid), str(area), ','.join(tile_ids)))

            # add geometries in the geometry dictionary nested in tile dictionary
            for tile_id in tile_ids:
                tile_dict[tile_id][orig_id].update({'geom': geom.ExportToWkt(),
                                                    'fid': fid})

            # update the dictionary of all features with the dictionary
            feat_dict.update({fid: {'geom': geom.ExportToWkt(),
                                    'orig_id': orig_id,
                                    'filename': filename,
                                    'area': area,
                                    'tiles': tile_ids}})
            feat_count += 1

    print('Assign all geometries to tiles...completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    nfeat = len(feat_dict)
    can_vec = None

    # create empty feature intersection list, empty list for each feature in the list
    feat_intersect_list = list([] for _ in range(0, nfeat))

    tt = time.time()

    gc.collect()

    # loop through the list of tiles by tile names
    for j, tile_name in enumerate(all_tile_names):
        wktlist = list()
        fid_list = list()

        # if tile exists in the tile dictionary, then make a list of all geometries in it
        if tile_name in tile_dict:

            # get all geometries in the tile
            for orig_id, geom_dict in tile_dict[tile_name].items():
                fid_list.append(geom_dict['fid'])
                wktlist.append(geom_dict['geom'])

            # make list of all geometries in the tile, and include
            # all their fid(s) from the feature list
            geom_list = list((fid_list[ij],
                              geom_wkt,
                              wktlist)
                             for ij, geom_wkt in enumerate(wktlist))

            # find all the intersecting/touching geometries
            # in the tile list using multi processing
            # add the intersection results to each feature
            # in the feature intersection list
            for fid, feat_intersecting in pool.imap_unordered(find_intersecting, geom_list):
                feat_intersect_list[fid] = feat_intersect_list[fid] + list(fid_list[ii] for ii in feat_intersecting)

            time_taken = Timer.display_time(time.time() - tt)

            print('Completed intersecting/touching features in tile {} : {} of {} in {}'.format(tile_name,
                                                                                                str(j+1),
                                                                                                len(all_tile_names),
                                                                                                time_taken))
            tt = time.time()
            sys.stdout.flush()

    # remove duplicates in each feature intersection list
    feat_intersect_list = list(list(set(temp_list)) for temp_list in feat_intersect_list)

    print('Feature intersection list...completed at {}'.format(Timer.display_time(time.time() - t)))
    sys.stdout.flush()

    # find all the geometries that do not intersect in the feature intersection list
    single_geom_fids = list(i for i in range(0, nfeat) if len(feat_intersect_list[i]) == 1)
    single_features = list(feat_dict[i] for i in single_geom_fids)

    # find features that intersect/touch other geometries
    multi_geom_fids = list(i for i in range(0, nfeat) if len(feat_intersect_list[i]) > 1)
    multi_geom_lists_scattered = list(feat_intersect_list[fid] for fid in multi_geom_fids)

    print('Starting grouping...')
    print('Length of scattered list: {}'.format(str(len(multi_geom_lists_scattered))))
    sys.stdout.flush()

    pickle_file = out_dir + 'temp_multi_geom_lists_scattered.pickle'

    with open(pickle_file, 'wb') as fileptr:
        pickle.dump(multi_geom_lists_scattered, fileptr)

    """
    # with open(pickle_file, 'rb') as fileptr:
        pickle.load(multi_geom_lists_scattered, fileptr)
    """
    part_extra = len(multi_geom_lists_scattered) % size - 1
    part_bins = list(range(0, len(multi_geom_lists_scattered), len(multi_geom_lists_scattered)//size))
    part_bins[-1] += part_extra
    part_tup_list = list((part_bins[i], part_bins[i+1]) for i in range(len(part_bins)-1))

    print(part_tup_list)

    multi_geom_lists_scattered_parts = list(list(multi_geom_lists_scattered[ii] for ii in range(jj, kk))
                                            for jj, kk in part_tup_list if jj < kk)

    multi_geom_lists_less_scattered = []
    for elem in pool.imap(group_multi, multi_geom_lists_scattered_parts):
        multi_geom_lists_less_scattered += elem

    # group all the fid(s) that occur together anywhere
    multi_geom_lists_grouped = group_multi(multi_geom_lists_less_scattered)

    print('Length of grouped list: {}'.format(str(len(multi_geom_lists_grouped))))

    print('Grouping multi geometry lists...completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()
    gc.collect()

    multi_features = multi_feat_union(multi_geom_lists_grouped)

    print('Merging geometries...completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()
    gc.collect()

    all_features = single_features + multi_features

    for key, _ in tile_dict.items():
        tile_dict[key] = dict()

    max_tiles = 1
    multi_tile_features = dict()
    for feat in all_features:
        tile_ids = feat['tiles']
        orig_id = feat['orig_id']
        for tile_id in tile_ids:
            tile_dict[tile_id].update({orig_id: feat})

        if len(tile_ids) > 1:
            multi_tile_features.update({orig_id: feat})
            if max_tiles < len(tile_ids):
                max_tiles = len(tile_ids)

    print('Assign geometries to tiles...completed at {}'.format(Timer.display_time(time.time() - t)))

    for tile_name in all_tile_names:
        tile_lakes = Vector(in_memory=True,
                            spref_str=main_crs_str,
                            geom_type=Vector.ogr_geom_type('polygon'))
        tile_lakes.name = tile_name
        tile_lakes.layer.CreateField(filename_attr)
        tile_lakes.layer.CreateField(id_attr)
        tile_lakes.layer.CreateField(area_attr)

        tile_lakes.fields = tile_lakes.fields + [filename_attr, id_attr, area_attr]
        tile_lakes.spref_str = main_crs_str

        if tile_name in tile_dict:
            for key, val in tile_dict[tile_name].items():
                lake_geom = ogr.CreateGeometryFromWkt(val['geom'])
                geom_area = val['area']
                lake_attr = {'orig_id': key,
                             'filename': val['filename'],
                             'area': geom_area}

                tile_lakes.add_feat(geom=lake_geom,
                                    attr=lake_attr)
        if tile_lakes.nfeat > 0:
            print('Writing vector: {}'.format(tile_lakes))
            tile_lakes.write_vector(out_dir + tile_name + '.shp')

    print('Step (write individual tile files) completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    print('Writing multi tile lakes...')

    multi_tile_lakes = Vector(in_memory=True,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))

    ntiles = ogr.FieldDefn('n_tiles', ogr.OFTInteger)
    ntiles.SetPrecision(9)

    multi_tile_lakes.name = 'multi_tile_lakes'
    multi_tile_lakes.layer.CreateField(filename_attr)
    multi_tile_lakes.layer.CreateField(id_attr)
    multi_tile_lakes.layer.CreateField(ntiles)
    multi_tile_lakes.layer.CreateField(area_attr)

    multi_tile_lakes.fields = multi_tile_lakes.fields + [filename_attr, id_attr, ntiles, area_attr]

    tile_attrs = list()
    for ti in range(max_tiles):
        tile_attr = ogr.FieldDefn('tile{}'.format(str(ti+1)), ogr.OFTString)
        multi_tile_lakes.layer.CreateField(tile_attr)
        multi_tile_lakes.fields.append(tile_attr)

    multi_tile_lakes.spref_str = main_crs_str

    for key, attr in multi_tile_features.items():
        geom = ogr.CreateGeometryFromWkt(attr['geom'])
        tile_list = attr['tiles']
        ntiles = len(tile_list)
        area = attr['area']

        attributes = {'orig_id': key,
                      'filename': attr['filename'],
                      'n_tiles': ntiles,
                      'area': area}

        for ti, tile in enumerate(tile_list):
            attributes.update({'tile{}'.format(str(ti+1)): tile})

        multi_tile_lakes.add_feat(geom=geom,
                                  attr=attributes)

    print(multi_tile_lakes)
    multi_tile_lakes.write_vector(out_dir + 'multi_tile_lakes.shp')

    print('Completed writing multi tile lakes at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    all_lakes_dict = dict(list((feat_dict['orig_id'], feat_dict) for feat_dict in all_features))

    print('Writing all lakes vector for Canada and Alaska...')
    all_lakes = Vector(in_memory=True,
                       spref_str=main_crs_str,
                       geom_type=Vector.ogr_geom_type('polygon'))

    ntiles = ogr.FieldDefn('n_tiles', ogr.OFTInteger)
    ntiles.SetPrecision(9)

    all_lakes.name = 'all_lakes'
    all_lakes.layer.CreateField(filename_attr)
    all_lakes.layer.CreateField(id_attr)
    all_lakes.layer.CreateField(ntiles)
    all_lakes.layer.CreateField(area_attr)

    all_lakes.fields = all_lakes.fields + [filename_attr, id_attr, ntiles, area_attr]

    tile_attrs = list()
    for ti in range(max_tiles):
        tile_attr = ogr.FieldDefn('tile{}'.format(str(ti+1)), ogr.OFTString)
        all_lakes.layer.CreateField(tile_attr)
        all_lakes.fields.append(tile_attr)

    all_lakes.spref_str = main_crs_str

    for key, attr in all_lakes_dict.items():
        geom = ogr.CreateGeometryFromWkt(attr['geom'])
        tile_list = attr['tiles']
        ntiles = len(tile_list)
        area = attr['area']

        attributes = {'orig_id': key,
                      'filename': attr['filename'],
                      'n_tiles': ntiles,
                      'area': area}

        for ti, tile in enumerate(tile_list):
            attributes.update({'tile{}'.format(str(ti+1)): tile})

        all_lakes.add_feat(geom=geom,
                           attr=attributes)

    print(all_lakes)
    all_lakes.write_vector(out_file)

    prj_file = out_file.split('.')[0] + '.prj'
    if File(prj_file).file_exists():
        File(prj_file).file_delete()

    with open(prj_file, 'w') as pf:
        pf.write(main_crs_str)

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

