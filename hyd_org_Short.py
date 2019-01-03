from osgeo import osr, ogr
from common import Vector, Timer, File
import multiprocessing as mp
import json
import sys
import time
import copy


# buffer distance for geometry touching/intersecting check
buffer_dist = 1  # meters


def find_tile(args):

    """
    Method to extract the tile number of each of the lakes
    :param args: Arguments
        (fid, feat_wkt, tile_wkts, tile_attrs)
    :return: tuple
    """

    fid, feat_wkt, tile_wkts, tile_attrs = args

    geom = ogr.CreateGeometryFromWkt(feat_wkt)

    tile_list = list()
    for ti, tile_wkt in enumerate(tile_wkts):
        if geom.Intersects(ogr.CreateGeometryFromWkt(tile_wkt)):
            tile_list.append(tile_attrs[ti]['grid_id'])

    return fid, tile_list


def find_intersecting(args):
    """
    Method to find intersecting lakes
    :param args: Arguments
        (fid, geom_wkt, wkt_list)
    :return: tuple
    """

    fid, geom_wkt, wkt_list = args

    geom = ogr.CreateGeometryFromWkt(geom_wkt).Buffer(buffer_dist)

    intersect_list = list()
    for ii, lake_wkt in enumerate(wkt_list):
        temp_geom = ogr.CreateGeometryFromWkt(lake_wkt)
        if geom.Intersects(temp_geom.Buffer(buffer_dist)):
            intersect_list.append(ii)

    return fid, intersect_list


def group_multi(in_list):
    """
    Method to group all numbers that occur together in any piece-wise-list manner
    :param in_list: List of lists
    :return: list of lists
    """
    out_list = copy.deepcopy(in_list)

    for j, elem in enumerate(out_list[:-1]):
        if len(elem) > 0:
            i = 0
            while i < len(elem):
                for check_elem in out_list[(j+1):]:
                    if len(check_elem) > 0:
                        if elem[i] in check_elem:
                            elem = elem + check_elem
                            idx_gen = sorted(list(range(len(check_elem))),
                                             reverse=True)
                            for idx in idx_gen:
                                check_elem.pop(idx)
                i += 1
            out_list[j] = sorted(list(set(elem)))

    empty_idx = sorted(list(i for i, elem in enumerate(out_list) if len(elem) == 0),
                       reverse=True)

    for idx in empty_idx:
        out_list.pop(idx)

    return out_list


if __name__ == '__main__':
    t = time.time()

    size = mp.cpu_count()
    pool = mp.Pool(processes=size)

    print('CPUs: {}'.format(str(size)))

    # file paths
    ak_file = "f:/hydroFlat/vectors/US_AK/3_merged/NHD_AK_WB_noGlac_diss_gte1000m2.shp"
    can_file_dir = "f:/hydroFlat/vectors/CAN/2_shps/indiv/"
    tile_file = "f:/hydroFlat/grids/PCS_NAD83_C_grid_ABoVE_intersection.shp"

    pre_merge_border_vec = "f:/hydroFlat/vectors/pre_merge_border_lakes2.shp"
    post_merge_border_vec = "f:/hydroFlat/vectors/post_merge_border_lakes2.shp"

    out_file = "f:/hydroFlat/vectors/alaska_canada_lakes2.json"

    out_dir = "f:/hydroFlat/lakes2/"

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

    main_crs_str = tile_vec.crs_string

    print(tile_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing ABoVE boundary vector.....')

    above_coords = [[[-168.83884, 66.60503], [-168.66305, 64.72256], [-166.11423, 63.29787], [-168.83884, 60.31062],
                     [-166.02634, 56.92698], [-166.64157, 54.70557], [-164.84625, 54.05535], [-157.94684, 54.69525],
                     [-153.64020, 56.21509], [-151.17926, 57.48851], [-149.64118, 58.87838], [-147.67361, 61.37118],
                     [-142.04861, 59.70736], [-135.67654, 58.69490], [-130.48731, 55.73262], [-124.82205, 50.42354],
                     [-113.70389, 51.06312], [-112.07791, 53.29901], [-109.00174, 53.03557], [-105.16527, 52.53873],
                     [-101.13553, 50.36751], [-98.007415, 49.77869], [-96.880859, 48.80976], [-94.983189, 48.94521],
                     [-94.851353, 52.79709], [-88.238500, 56.92737], [-91.862463, 57.81702], [-93.775610, 59.60700],
                     [-92.984594, 61.25472], [-87.315649, 64.30688], [-80.504125, 66.77919], [-79.976781, 68.59675],
                     [-81.426977, 69.84364], [-84.547094, 70.00956], [-87.447485, 69.93430], [-91.094946, 70.77629],
                     [-91.798071, 72.17192], [-89.688696, 73.86475], [-89.600805, 74.33426], [-92.940649, 74.61654],
                     [-93.380102, 75.58784], [-94.874242, 75.69681], [-95.137914, 75.86949], [-96.719946, 76.56045],
                     [-97.598852, 76.81343], [-97.618407, 77.32284], [-99.552001, 78.91297], [-103.94653, 79.75829],
                     [-113.79028, 78.81110], [-124.33715, 76.52777], [-128.02856, 71.03224], [-136.99340, 69.67342],
                     [-149.64965, 71.03224], [-158.08715, 71.65080], [-167.93090, 69.24910]]]

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
    print('processing border vector.....')

    # make border vector in memory
    border_json_str = json.dumps({"type": "Polygon", "coordinates": [[[-142.277, 70.526], [-141.749, 60.199],
                                                                      [-130.675, 54.165], [-128.126, 55.680],
                                                                      [-135.421, 60.111], [-137.355, 59.715],
                                                                      [-139.904, 61.061], [-140.167, 70.438]]]})

    border_geom = ogr.CreateGeometryFromJson(border_json_str)
    border_geom.CloseRings()
    border_geom = Vector.reproj_geom(border_geom,
                                     geog_spref.ExportToWkt(),
                                     main_crs_str)

    border_vec = Vector(in_memory=True,
                        spref_str=geog_spref.ExportToWkt(),
                        geom_type=Vector.ogr_geom_type('polygon'))

    border_vec.name = 'border'
    border_vec.add_feat(border_geom)

    print(border_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing ak vector.....')

    sys.stdout.flush()

    ak_vec = Vector(in_memory=True,
                    spref_str=main_crs_str,
                    geom_type=Vector.ogr_geom_type('polygon'))

    ak_vec.name = 'alaska_lakes'

    ak_vec.layer.CreateField(filename_attr)
    ak_vec.layer.CreateField(id_attr)
    ak_vec.fields = ak_vec.fields + [filename_attr, id_attr]
    ak_vec.crs_string = main_crs_str

    temp_vec = None
    geom_list = list()

    filename = File(ak_file).basename.split('.shp')[0]

    try:
        temp_vec = Vector(ak_file,
                          spref_str=main_crs_str,
                          geom_type=Vector.ogr_geom_type('polygon'),
                          feat_limit=200)

    except (AttributeError, RuntimeError):
        error_list.append('Error reading {}'.format(ak_file))
        print('Error reading {}'.format(ak_file))

    if temp_vec:
        if temp_vec.bounds.Intersects(above_geom):
            print(temp_vec)

            geom_list = list()
            for feat in temp_vec.features:
                geom = feat.GetGeometryRef()
                geom_dict = feat.items()

                geom_list.append((str(geom_dict['OBJECTID']),
                                  geom.ExportToWkt(),
                                  tile_vec.wkt_list,
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
    can_vec.crs_string = main_crs_str

    for i, can_file in enumerate(can_file_list[491:495]):
        filename = File(can_file).basename.split('.shp')[0]
        geom_list = list()

        print('File {} of {}: {}'.format(str(i+1),
                                         str(len(can_file_list)),
                                         can_file))
        temp_vec = None

        try:
            temp_vec = Vector(filename=can_file,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'),
                              feat_limit=200)

        except (AttributeError, RuntimeError):
            error_list.append('Error reading {}'.format(can_file))
            print('Error reading {}'.format(can_file))

        if temp_vec:
            if temp_vec.bounds.Intersects(above_geom):
                print(temp_vec)

                geom_list = list()
                for feat in temp_vec.features:
                    geom = feat.GetGeometryRef()
                    geom_dict = feat.items()

                    geom_list.append((str(geom_dict['NID']),
                                      geom.ExportToWkt(),
                                      tile_vec.wkt_list,
                                      tile_vec.attributes))

                    can_vec.add_feat(geom=geom,
                                     attr={'filename': filename,
                                           'orig_id': geom_dict['NID']})

                results = pool.map(find_tile,
                                   geom_list)

                can_vec.data.update(dict(results))
            else:
                print('Vector is outside AOI')

            temp_vec = None

        sys.stdout.flush()

    print(can_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

    for err in error_list:
        print(err)

    print('----------------------------------------------------------------')
    print('Spatial ref strings: \n')
    print(ak_vec.crs_string)
    print(can_vec.crs_string)
    print(tile_vec.crs_string)
    print('----------------------------------------------------------------')
    print('Merge Canada and Alaska lakes...')
    sys.stdout.flush()

    can_vec.merge(ak_vec, True)

    can_vec.name = 'ak_can_merged'
    print('Merged initial vector: {}'.format(can_vec))

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    all_tile_names = list(attr['grid_id'] for attr in tile_vec.attributes)
    print('Get all tile names...completed at {}'.format(Timer.display_time(time.time() - t)))

    tile_dict = dict()
    for key, vals in can_vec.data.items():
        for val in vals:
            if val not in tile_dict:
                tile_dict[val] = dict()
            tile_dict[val].update({key: dict()})
    print('Dictionary structure to store lakes by tile...completed at {}'.format(Timer.display_time(time.time() - t)))

    feat_dict = dict()
    feat_count = 0
    for feat in can_vec.features:
        geom = feat.GetGeometryRef()
        orig_id = feat.items()['orig_id']
        area = geom.GetArea()

        tile_ids = can_vec.data[orig_id]

        if len(tile_ids) > 0:
            fid = feat_count
            # print('ID: {} Area: {} Tiles: {}'.format(str(fid), str(area), ','.join(tile_ids)))

            for tile_id in tile_ids:
                tile_dict[tile_id][orig_id].update({'geom': geom.ExportToWkt(),
                                                    'fid': fid})

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

    feat_intersect_list = list([] for _ in range(0, nfeat))

    for tile_name in all_tile_names:
        wkt_list = list()
        fid_list = list()

        if tile_name in tile_dict:
            print('Finding intersecting features in tile {}'.format(tile_name))

            for orig_id, geom_dict in tile_dict[tile_name].items():
                fid_list.append(geom_dict['fid'])
                wkt_list.append(geom_dict['geom'])

            geom_list = list((fid_list[ij],
                              geom_wkt,
                              wkt_list)
                             for ij, geom_wkt in enumerate(wkt_list))

            intersect_results = pool.map(find_intersecting,
                                         geom_list)

            for fid, feat_intersects in intersect_results:
                feat_intersect_list[fid] = feat_intersect_list[fid] + list(fid_list[ii] for ii in feat_intersects)

    feat_intersect_list = list(list(set(temp_list)) for temp_list in feat_intersect_list)

    print('Feature intersection list...completed at {}'.format(Timer.display_time(time.time() - t)))

    single_geom_fids = list(i for i in range(0, nfeat) if len(feat_intersect_list[i]) == 1)
    single_features = list(feat_dict[i] for i in single_geom_fids)

    multi_geom_fids = list(i for i in range(0, nfeat) if len(feat_intersect_list[i]) > 1)
    multi_geom_lists_scattered = list(feat_intersect_list[fid] for fid in multi_geom_fids)

    multi_geom_lists_grouped = group_multi(multi_geom_lists_scattered)

    print('Grouping multi geometry lists...completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    multi_features = list()
    for multi_list in multi_geom_lists_grouped:
        features = sorted(list(feat_dict[i] for i in multi_list),
                          key=lambda k: k['area'],
                          reverse=True)
        feat_id = features[0]['orig_id']
        feat_filename = features[0]['filename']

        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        tiles = list()
        for feat in features:
            geom = ogr.CreateGeometryFromWkt(feat['geom'])
            multi_geom.AddGeometryDirectly(geom.Buffer(buffer_dist))
            tiles = tiles + feat['tiles']

        tiles = list(set(tiles))

        grp_geom = multi_geom.UnionCascaded()
        area = grp_geom.GetArea()

        multi_features.append({'geom': grp_geom.ExportToWkt(),
                               'filename': feat_filename,
                               'orig_id': feat_id,
                               'area': area,
                               'tiles': tiles})

    print('Merging geometries...completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

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
        tile_lakes.crs_string = main_crs_str

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

    multi_tile_lakes.crs_string = main_crs_str

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

    all_lakes.crs_string = main_crs_str

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

