from osgeo import osr, ogr
from common import Vector, Timer, File
import multiprocessing as mp
import json
import sys
import time
import copy


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

    geom = ogr.CreateGeometryFromWkt(geom_wkt)

    intersect_list = list()
    for ii, lake_wkt in enumerate(wkt_list):
        temp_geom = ogr.CreateGeometryFromWkt(lake_wkt)
        if geom.Intersects(temp_geom.Buffer(1)):
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

    # file paths
    ak_file = "f:/hydroFlat/vectors/US_AK/3_merged/NHD_AK_WB_noGlac_diss_gte1000m2.shp"
    can_file_dir = "f:/hydroFlat/vectors/CAN/2_shps/indiv/"
    tile_file = "f:/hydroFlat/grids/PCS_NAD83_C_grid_ABoVE_intersection.shp"

    pre_merge_border_vec = "f:/hydroFlat/vectors/pre_merge_border_lakes2.shp"
    post_merge_border_vec = "f:/hydroFlat/vectors/post_merge_border_lakes2.shp"

    out_file = "f:/hydroFlat/vectors/alaska_canada_lakes2.json"

    out_dir = "f:/hydroFlat/lakes2/"

    buffer_dist = 1  # meters

    error_list = list()
    border_list = list()

    geog_spref = osr.SpatialReference()
    geog_spref.ImportFromProj4('+proj=longlat +datum=WGS84')

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

    filename_attr = ogr.FieldDefn('filename', ogr.OFTString)
    id_attr = ogr.FieldDefn('orig_id', ogr.OFTString)
    area_attr = ogr.FieldDefn('area', ogr.OFTReal)
    area_attr.SetPrecision(8)
    area_attr.SetWidth(32)

    ak_vec.layer.CreateField(filename_attr)
    ak_vec.layer.CreateField(id_attr)
    ak_vec.fields = ak_vec.fields + [filename_attr, id_attr]
    ak_vec.crs_string = main_crs_str

    temp_vec = None
    samp_list = list()

    try:
        temp_vec = Vector(ak_file,
                          spref_str=main_crs_str,
                          geom_type=Vector.ogr_geom_type('polygon'))

    except (AttributeError, RuntimeError):
        error_list.append('Error reading {}'.format(ak_file))
        print('Error reading {}'.format(ak_file))

    if temp_vec:
        filename = File(ak_file).basename.split('.shp')[0]

        for feat in temp_vec.features:
            geom = feat.GetGeometryRef()
            orig_id = str(feat.items()['OBJECTID'])

            if geom.Intersects(border_geom):
                border_list.append({'orig_id': orig_id,
                                    'filename': filename,
                                    'geom': geom.ExportToWkt()})
            else:
                samp_list.append((orig_id,
                                  geom.ExportToWkt(),
                                  tile_vec.wkt_list,
                                  tile_vec.attributes))

                ak_vec.add_feat(geom=geom,
                                attr={'filename': filename,
                                      'orig_id': orig_id})

        temp_vec = None

    print('Finding tile names for ak vector...')

    sys.stdout.flush()

    results = pool.map(find_tile,
                       samp_list)

    ak_vec.data = dict(results)

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

    for i, can_file in enumerate(can_file_list):
        filename = File(can_file).basename.split('.shp')[0]
        samp_list = list()

        print('File {} of {}: {}'.format(str(i+1),
                                         str(len(can_file_list)),
                                         can_file))
        temp_vec = None

        try:
            temp_vec = Vector(filename=can_file,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))

        except (AttributeError, RuntimeError):
            error_list.append('Error reading {}'.format(can_file))
            print('Error reading {}'.format(can_file))

        if temp_vec:
            if temp_vec.bounds.Intersects(above_geom):

                geom_sample_list = list()

                for ij in range(0, temp_vec.nfeat):
                    geom = ogr.CreateGeometryFromWkt(temp_vec.wkt_list[ij])
                    geom_buf = geom.Buffer(buffer_dist)
                    geom_wkt = geom_buf.ExportToWkt()

                    geom_sample_list.append((ij,
                                             geom_wkt,
                                             temp_vec.wkt_list))

                intersect_results = dict(pool.map(find_intersecting,
                                                  geom_sample_list))

                multi_list = list()
                samp_list = list()
                for ig in range(0, temp_vec.nfeat):
                    feat = temp_vec.features[ig]
                    if len(intersect_results[ig]) == 1:
                        geom = feat.GetGeometryRef()
                        geom_dict = temp_vec.attributes[ig]

                        if geom.Intersects(border_geom):
                            border_list.append({'orig_id': geom_dict['NID'],
                                                'filename': filename,
                                                'geom': temp_vec.wkt_list[ig]})
                        else:
                            samp_list.append((geom_dict['NID'],
                                              temp_vec.wkt_list[ig],
                                              tile_vec.wkt_list,
                                              tile_vec.attributes))

                            can_vec.add_feat(geom=geom,
                                             attr={'filename': filename,
                                                   'orig_id': geom_dict['NID']})

                    elif len(intersect_results[ig]) > 1:
                        multi_list.append(intersect_results[ig])

                print('Found {} non-intersecting lakes for {}'.format(str(len(samp_list)),
                                                                      temp_vec.name))

                print('Found {} intersecting/touching lakes for {}'.format(str(len(multi_list)),
                                                                           temp_vec.name))

                print('Finding tile names for non-intersecting geometries...')

                single_geom_results = pool.map(find_tile,
                                               samp_list)

                can_vec.data.update(dict(single_geom_results))

                if len(multi_list) > 0:

                    print('Grouping multiple geometries...')

                    multi_list = group_multi(multi_list)

                    samp_list = list()
                    for grp in multi_list:
                        grp_orig_id = temp_vec.attributes[grp[0]]['NID']
                        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)

                        for elem in grp:
                            geom = ogr.CreateGeometryFromWkt(temp_vec.wkt_list[elem])
                            multi_geom.AddGeometryDirectly(geom.Buffer(buffer_dist))

                        grp_geom = multi_geom.UnionCascaded()

                        if grp_geom.Intersects(border_geom):
                            border_list.append({'orig_id': grp_orig_id,
                                                'filename': filename,
                                                'geom': grp_geom.ExportToWkt()})
                        else:

                            samp_list.append((grp_orig_id,
                                              grp_geom.ExportToWkt(),
                                              tile_vec.wkt_list,
                                              tile_vec.attributes))

                            can_vec.add_feat(geom=grp_geom,
                                             attr={'filename': filename,
                                                   'orig_id': grp_orig_id})

                    print('Finding tile names for grouped intersecting geometries...')

                    multi_geom_results = pool.map(find_tile,
                                                  samp_list)

                    can_vec.data.update(dict(multi_geom_results))
            else:
                print('Vector is outside AOI')

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
    print(can_vec)

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Output unmerged lakes in border region...')
    sys.stdout.flush()

    border_lakes_pre = Vector(in_memory=True,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))
    border_lakes_pre.name = 'border_lakes_pre'
    border_lakes_pre.layer.CreateField(filename_attr)
    border_lakes_pre.layer.CreateField(id_attr)
    border_lakes_pre.fields = border_lakes_pre.fields + [filename_attr, id_attr]
    border_lakes_pre.crs_string = main_crs_str

    for lake in border_list:
        border_lakes_pre.add_feat(geom=ogr.CreateGeometryFromWkt(lake['geom']),
                                  attr={'orig_id': lake['orig_id'],
                                        'filename': lake['filename']})

    border_lakes_pre.write_vector(pre_merge_border_vec)

    print('Lakes in border region: {}'.format(len(border_list)))
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Rearranging and merging border lakes:')
    sys.stdout.flush()
    sample_list = list()

    for i in range(0, len(border_list)):
        geom = ogr.CreateGeometryFromWkt(border_list[i]['geom'])
        geom_buf = geom.Buffer(buffer_dist)
        geom_wkt = geom_buf.ExportToWkt()

        sample_list.append((i,
                            geom_wkt,
                            border_lakes_pre.wkt_list))

    intersect_results = dict(pool.map(find_intersecting,
                                      sample_list))

    intersect_list = list(intersect_results[ii] for ii in range(0, len(border_list)))

    multi_list = list()
    samp_list = list()
    for geom_list in intersect_list:
        if len(geom_list) == 1:
            geom_dict = border_list[geom_list[0]]

            samp_list.append((geom_dict['orig_id'],
                              geom_dict['geom'],
                              tile_vec.wkt_list,
                              tile_vec.attributes))

            can_vec.add_feat(geom=ogr.CreateGeometryFromWkt(geom_dict['geom']),
                             attr={'filename': geom_dict['filename'],
                                   'orig_id': geom_dict['orig_id']})

        elif len(geom_list) > 1:
            multi_list.append(geom_list)

    print('Found {} intersecting lakes at the border'.format(str(len(multi_list))))

    single_geom_results = pool.map(find_tile,
                                   samp_list)

    can_vec.data.update(dict(single_geom_results))

    print('Step (find multi intersecting) completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    multi_list = group_multi(multi_list)

    print('Final list of intersecting lakes has {} elements'.format(str(len(multi_list))))

    print('Step (group multi intersecting) completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Merging geometries...')
    sys.stdout.flush()

    border_lakes_post = Vector(in_memory=True,
                               spref_str=main_crs_str,
                               geom_type=Vector.ogr_geom_type('polygon'))

    border_lakes_post.name = 'border_lakes_post'
    border_lakes_post.layer.CreateField(filename_attr)
    border_lakes_post.layer.CreateField(id_attr)

    border_lakes_post.fields = border_lakes_post.fields + [filename_attr, id_attr]
    border_lakes_post.crs_string = main_crs_str

    sample_list = list()

    for grp in multi_list:
        print(grp)
        grp_orig_id = border_list[grp[0]]['orig_id']
        grp_filename = border_list[grp[0]]['filename']

        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        for elem in grp:
            geom = ogr.CreateGeometryFromWkt(border_list[elem]['geom'])
            multi_geom.AddGeometryDirectly(geom.Buffer(buffer_dist))

        grp_geom = multi_geom.UnionCascaded()

        sample_list.append((grp_orig_id,
                            grp_geom.ExportToWkt(),
                            tile_vec.wkt_list,
                            tile_vec.attributes))

        can_vec.add_feat(geom=grp_geom,
                         attr={'filename': grp_filename,
                               'orig_id': grp_orig_id})

        border_lakes_post.add_feat(geom=grp_geom,
                                   attr={'filename': grp_filename,
                                         'orig_id': grp_orig_id})

    multi_geom_results = pool.map(find_tile,
                                  sample_list)

    can_vec.data.update(dict(multi_geom_results))

    border_lakes_post.write_vector(post_merge_border_vec)

    print('Step (merge border lakes) completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

    border_lakes_pre = border_lakes_post = None

    print('Final vector: {}'.format(can_vec))
    sys.stdout.flush()

    print('Writing individual tile files...')

    all_tile_names = list(attr['grid_id'] for attr in tile_vec.attributes)
    print('Get all tile names...completed at {}'.format(Timer.display_time(time.time() - t)))

    tile_dict = dict()
    for key, vals in can_vec.data.items():
        for val in vals:
            if val not in tile_dict:
                tile_dict[val] = dict()
            tile_dict[val].update({key: dict()})
    print('Dictionary structure to store lakes by tile...completed at {}'.format(Timer.display_time(time.time() - t)))

    multi_tile_dict = dict()
    max_tiles = 1
    for lake_feat in can_vec.features:
        geom = lake_feat.GetGeometryRef()
        orig_id = lake_feat.items()['orig_id']
        filename = lake_feat.items()['filename']

        lake_tiles = can_vec.data[orig_id]

        for lake_tile in lake_tiles:
            tile_dict[lake_tile][orig_id].update({'geom': geom.ExportToWkt(),
                                                  'filename': filename})

        if len(lake_tiles) > 1:
            if len(lake_tiles) > max_tiles:
                max_tiles = len(lake_tiles)
            multi_tile_dict.update({orig_id: {'geom': geom.ExportToWkt(),
                                              'filename': filename,
                                              'tiles': lake_tiles}})
    print('Extract lake parameters...completed at {}'.format(Timer.display_time(time.time() - t)))

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
                geom_area = lake_geom.GetArea()
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

    for key, attr in multi_tile_dict.items():
        geom = ogr.CreateGeometryFromWkt(attr['geom'])
        tile_list = attr['tiles']
        ntiles = len(tile_list)
        area = geom.GetArea()

        attributes = {'orig_id': key,
                      'filename': attr['filename'],
                      'n_tiles': ntiles,
                      'area': area}

        for t, tile in enumerate(tile_list):
            attributes.update({'tile{}'.format(str(t+1)): tile})

        multi_tile_lakes.add_feat(geom=geom,
                                  attr=attributes)

    print(multi_tile_lakes)
    multi_tile_lakes.write_vector(out_dir + 'multi_tile_lakes.shp')

    print('Completed writing multi tile lakes at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    print('Writing all lakes vector for Canada and Alaska...')
    can_vec.write_vector(outfile=out_file)

    prj_file = out_file.split('.')[0] + '.prj'
    if File(prj_file).file_exists():
        File(prj_file).file_delete()

    with open(prj_file, 'w') as pf:
        pf.write(can_vec.spref.ExportToWkt())

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

