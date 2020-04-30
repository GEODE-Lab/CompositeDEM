from osgeo import osr, ogr
from demLib.common import Vector, Timer, File
import multiprocessing as mp
import json
import sys
import time


def find_tile(args_list):

    """
    Method to extract the tile number of each of the lakes
    :param args_list: Arguments
        feat_wkt, feat_dict, tile_wkts, tile_attrs, filename
    :return: dictionary
    """
    out_list = list()

    for args in args_list:

        fid, feat_wkt, tile_wkts, tile_attrs= args

        geom = ogr.CreateGeometryFromWkt(feat_wkt)

        tile_list = list()
        for ti, tile_wkt in enumerate(tile_wkts):
            if geom.Intersects(ogr.CreateGeometryFromWkt(tile_wkt)):
                tile_list.append(tile_attrs[ti]['grid_id'])

        out_list.append((fid, tile_list))

    return out_list


if __name__ == '__main__':
    t = time.time()

    size = mp.cpu_count()

    # file paths
    ak_file = "f:/hydroFlat/vectors/US_AK/3_merged/NHD_AK_WB_noGlac_diss_gte1000m2.shp"
    can_file_dir = "f:/hydroFlat/vectors/CAN/2_shps/indiv/"
    tile_file = "f:/hydroFlat/grids/PCS_NAD83_C_grid_ABoVE_intersection.shp"

    pre_merge_border_vec = "f:/hydroFlat/vectors/pre_merge_border_lakes.shp"
    post_merge_border_vec = "f:/hydroFlat/vectors/post_merge_border_lakes.shp"

    out_file = "f:/hydroFlat/vectors/alaska_canada_lakes.shp"

    out_dir = "f:/hydroFlat/lakes/"

    error_list = list()
    border_list = list()

    print('----------------------------------------------------------------')
    print('processing tile vector.....')

    # read tiles in memory
    tile_vec = Vector(filename=tile_file,
                      verbose=False)

    main_crs_str = tile_vec.spref_str

    print(tile_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing border vector.....')

    # make border vector in memory
    border_json = {"type": "Polygon", "coordinates": [[[-142.277, 70.526], [-141.749, 60.199], [-130.675, 54.165],
                                                       [-128.126, 55.680], [-135.421, 60.111], [-137.355, 59.715],
                                                       [-139.904, 61.061], [-140.167, 70.438]]]}

    border_spref = osr.SpatialReference()
    border_spref.ImportFromProj4('+proj=longlat +datum=WGS84')

    border_json_str = json.dumps(border_json)
    border_geom = ogr.CreateGeometryFromJson(border_json_str)
    border_geom.CloseRings()
    border_geom = Vector.reproj_geom(border_geom,
                                     border_spref.ExportToWkt(),
                                     main_crs_str)

    border_vec = Vector(in_memory=True,
                        spref_str=border_spref.ExportToWkt(),
                        geom_type=Vector.ogr_geom_type('polygon'))

    border_vec.name = 'border'
    border_vec.add_feat(border_geom)

    print(border_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('processing ak vector.....')

    ak_vec = Vector(in_memory=True,
                    spref_str=main_crs_str,
                    geom_type=Vector.ogr_geom_type('polygon'))

    ak_vec.name = 'alaska_lakes'

    filename_attr = ogr.FieldDefn('filename', ogr.OFTString)
    id_attr = ogr.FieldDefn('orig_id', ogr.OFTString)
    tile_attr = ogr.FieldDefn('tiles', ogr.OFTString)

    ak_vec.layer.CreateField(filename_attr)
    ak_vec.layer.CreateField(id_attr)
    ak_vec.layer.CreateField(tile_attr)
    ak_vec.fields = ak_vec.fields + [filename_attr, id_attr, tile_attr]
    ak_vec.crs_string = main_crs_str

    ak_count = 0
    temp_vec = None

    try:
        temp_vec = Vector(ak_file,
                          spref_str=main_crs_str,
                          geom_type=Vector.ogr_geom_type('polygon'))

    except (AttributeError, RuntimeError):
        error_list.append('Error reading {}'.format(ak_file))
        print('Error reading {}'.format(ak_file))

    if temp_vec:
        filename = File(ak_file).basename.split('.shp')[0]

        # multiprocessing -------------------------------------------------
        samp_list = list()
        for feat in temp_vec.features:
            geom = feat.GetGeometryRef()
            orig_id = str(feat.items()['OBJECTID'])

            samp_list.append((orig_id,
                              geom.ExportToWkt(),
                              tile_vec.wkt_list,
                              tile_vec.attributes))

        sys.stdout.write('Number of elements in {} sample list : {}\n'.format(filename,
                                                                              str(len(samp_list))))

        samp_chunks = [samp_list[i::size] for i in xrange(size)]
        chunk_length = list(str(len(chunk)) for chunk in samp_chunks)

        sys.stdout.write(' Distribution of chunks : {}\n'.format(', '.join(chunk_length)))
        sys.stdout.flush()

        pool = mp.Pool(processes=size)

        results = pool.map(find_tile,
                           samp_chunks)

        out_list = list()
        if len(results) > 0:
            for result in results:
                for vals in result:
                    out_list.append(vals)
        out_dict = dict(out_list)
        # multiprocessing -------------------------------------------------

        for feat in temp_vec.features:
            geom = feat.GetGeometryRef()
            orig_id = str(feat.items()['OBJECTID'])

            tile_list = out_dict[orig_id]

            if geom.Intersects(border_geom):
                border_list.append({'orig_id': orig_id,
                                    'filename': filename,
                                    'geom': geom,
                                    'tiles': tile_list})

            ak_vec.add_feat(geom=geom,
                            attr={'filename': filename,
                                  'orig_id': orig_id,
                                  'tiles': '-'.join(tile_list)})

        ak_count = ak_count + temp_vec.nfeat
        temp_vec = None

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
    can_vec.layer.CreateField(tile_attr)
    can_vec.fields = can_vec.fields + [filename_attr, id_attr, tile_attr]
    can_vec.crs_string = main_crs_str

    can_count = 0
    for i, can_file in enumerate(can_file_list):
        filename = File(can_file).basename.split('.shp')[0]

        print('File {} of {}: {}'.format(str(i+1),
                                         str(len(can_file_list)),
                                         can_file))
        temp_vec = None

        try:
            temp_vec = Vector(filename=can_file,
                              spref_str=main_crs_str)

        except (AttributeError, RuntimeError):
            error_list.append('Error reading {}'.format(can_file))
            print('Error reading {}'.format(can_file))
        if temp_vec:

            # multiprocessing -------------------------------------------------
            samp_list = list()
            for feat in temp_vec.features:
                geom = feat.GetGeometryRef()
                orig_id = str(feat.items()['NID'])

                samp_list.append((orig_id,
                                  geom.ExportToWkt(),
                                  tile_vec.wkt_list,
                                  tile_vec.attributes))

            sys.stdout.write('Number of elements in {} sample list : {}\n'.format(filename,
                                                                                str(len(samp_list))))

            samp_chunks = [samp_list[i::size] for i in xrange(size)]
            chunk_length = list(str(len(chunk)) for chunk in samp_chunks)

            sys.stdout.write(' Distribution of chunks : {}\n'.format(', '.join(chunk_length)))
            sys.stdout.flush()

            pool = mp.Pool(processes=size)

            results = pool.map(find_tile,
                               samp_chunks)

            out_list = list()
            if len(results) > 0:
                for result in results:
                    for vals in result:
                        out_list.append(vals)
            out_dict = dict(out_list)

            # multiprocessing -------------------------------------------------

            for feat in temp_vec.features:
                geom = feat.GetGeometryRef()
                orig_id = feat.items()['NID']

                tile_list = out_dict[orig_id]

                if geom.Intersects(border_geom):
                    border_list.append({'orig_id': orig_id,
                                        'filename': filename,
                                        'geom': geom,
                                        'tiles': tile_list})

                can_vec.data.append(tile_list)
                can_vec.add_feat(geom=geom,
                                 attr={'filename': filename,
                                       'orig_id': orig_id,
                                       'tiles': '-'.join(tile_list)})

            can_count = can_count + temp_vec.nfeat
            temp_vec = None

        sys.stdout.flush()

    print(can_vec)
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

    for error in error_list:
        print(error)
    sys.stdout.flush()
    print('----------------------------------------------------------------')
    print('Spatial ref strings: \n')
    print(ak_vec.crs_string)
    print(can_vec.crs_string)
    print(tile_vec.crs_string)
    print('----------------------------------------------------------------')
    print('Merge Canada and Alaska lakes...')
    can_vec.merge(ak_vec, True)

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Output unmerged lakes in border region...')

    border_lakes_pre = Vector(in_memory=True,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))
    border_lakes_pre.name = 'border_lakes_pre'
    border_lakes_pre.layer.CreateField(filename_attr)
    border_lakes_pre.layer.CreateField(id_attr)
    border_lakes_pre.layer.CreateField(tile_attr)
    border_lakes_pre.fields = border_lakes_pre.fields + [filename_attr, id_attr, tile_attr]
    border_lakes_pre.crs_string = main_crs_str

    for lake in border_list:
        border_lakes_pre.add_feat(geom=lake['geom'],
                                  attr={'orig_id': lake['orig_id'],
                                        'filename': lake['filename'],
                                        'tiles': lake['tiles']})

    border_lakes_pre.write_vector(pre_merge_border_vec)

    print('Lakes in border region: {}'.format(len(border_list)))
    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Removing border regions from lakes vector...')
    sys.stdout.flush()

    border_ids = list(elem_dict['orig_id'] for elem_dict in border_list)
    pop_list = list()

    for i, attr in enumerate(can_vec.attributes):
        if attr['orig_id'] in border_ids:
            pop_list.append(i)

    pop_list = sorted(pop_list,
                      reverse=True)

    for i in pop_list:
        can_vec.attributes.pop(i)
        can_vec.features.pop(i)
        can_vec.wkt_list.pop(i)
        can_vec.data.pop(i)

    can_vec.nfeat = len(can_vec.features)

    print(can_vec)

    sys.stdout.flush()

    buffer_dist = 1  # meters

    intersect_list = list()

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    print('Rearranging and merging border lakes:')
    sys.stdout.flush()

    for i in range(0, len(border_list)):
        geom_buf = border_list[i]['geom'].Buffer(buffer_dist)
        geom_list = list()

        for j in range(0, len(border_list)):
            temp_geom = border_list[j]['geom'].Buffer(buffer_dist)

            if geom_buf.Intersects(temp_geom):
                geom_list.append(j)

        intersect_list.append(geom_list)

    multi_list = list()

    for geom_list in intersect_list:
        if len(geom_list) == 1:
            geom_dict = border_list[geom_list[0]]
            can_vec.add_feat(geom=geom_dict['geom'],
                             attr={'filename': geom_dict['filename'],
                                   'orig_id': geom_dict['orig_id'],
                                   'tiles': geom_dict['tiles']})
        elif len(geom_list) > 1:
            multi_list.append(geom_list)

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    for j, multi in enumerate(multi_list[:-1]):
        if len(multi) > 0:
            i = 0
            while i < len(multi):
                for check_multi in multi_list[(j+1):]:
                    if len(check_multi) > 0:
                        if multi[i] in check_multi:
                            multi = multi + check_multi
                            index_gen = sorted(list(range(len(check_multi))),
                                               reverse=True)
                            for index in index_gen:
                                check_multi.pop(index)
                i += 1
            multi_list[j] = sorted(list(set(multi)))

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    empty_idx = sorted(list(i for i, elem in enumerate(multi_list) if len(elem) == 0),
                       reverse=True)

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    for idx in empty_idx:
        multi_list.pop(idx)

    border_lakes_post = Vector(in_memory=True,
                               spref_str=main_crs_str,
                               geom_type=Vector.ogr_geom_type('polygon'))
    border_lakes_post.name = 'border_lakes_post'
    border_lakes_post.layer.CreateField(filename_attr)
    border_lakes_post.layer.CreateField(id_attr)
    border_lakes_post.layer.CreateField(tile_attr)
    border_lakes_post.fields = border_lakes_post.fields + [filename_attr, id_attr, tile_attr]
    border_lakes_post.crs_string = main_crs_str

    for grp in multi_list:
        print(grp)
        grp_orig_id = border_list[grp[0]]['orig_id']
        grp_filename = border_list[grp[0]]['filename']

        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        for elem in grp:
            multi_geom.AddGeometryDirectly(border_list[elem]['geom'])

        grp_geom = multi_geom.UnionCascaded()

        tile_list = list()
        for ti, tile_feat in enumerate(tile_vec.features):
            if grp_geom.Intersects(tile_feat.GetGeometryRef()):
                tile_list.append(tile_vec.attributes[ti]['grid_id'])

        can_vec.add_feat(geom=grp_geom,
                         attr={'filename': grp_filename,
                               'orig_id': grp_orig_id,
                               'tiles': '-'.join(tile_list)})

        border_lakes_post.add_feat(geom=grp_geom,
                                   attr={'filename': grp_filename,
                                         'orig_id': grp_orig_id,
                                         'tiles': '-'.join(tile_list)})

    border_lakes_post.write_vector(post_merge_border_vec)

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    print('Writing individual tile files...')

    all_attr = can_vec.attributes

    all_tile_names = list(attr['grid_id'] for attr in tile_vec.attributes)

    for tile_name in all_tile_names:
        lake_indices = list(i for i, attr in enumerate(all_attr) if tile_name in attr['tiles'])

        tile_lakes = Vector(in_memory=True,
                            spref_str=main_crs_str,
                            geom_type=Vector.ogr_geom_type('polygon'))
        tile_lakes.name = tile_name
        tile_lakes.layer.CreateField(filename_attr)
        tile_lakes.layer.CreateField(id_attr)
        tile_lakes.layer.CreateField(tile_attr)
        tile_lakes.fields = tile_lakes.fields + [filename_attr, id_attr, tile_attr]
        tile_lakes.crs_string = main_crs_str

        for l, lake in enumerate(lake_indices):
            lake_geom = can_vec.features[l].GetGeometryRef()
            lake_attr = can_vec.attributes[l]

            tile_lakes.add_feat(geom=lake_geom,
                                attr=lake_attr)

        print(tile_lakes)
        tile_lakes.write_vector(out_dir + tile_name + '.shp')

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    print('Writing multi tile lakes...')

    multi_tile_lake_indices = list(i for i, attr in enumerate(all_attr) if '-' in attr['tiles'])

    multi_tile_lakes = Vector(in_memory=True,
                              spref_str=main_crs_str,
                              geom_type=Vector.ogr_geom_type('polygon'))
    multi_tile_lakes.name = 'multi_tile_lakes'

    multi_tile_lakes.layer.CreateField(filename_attr)
    multi_tile_lakes.layer.CreateField(id_attr)
    multi_tile_lakes.layer.CreateField(tile_attr)
    multi_tile_lakes.fields = multi_tile_lakes.fields + [filename_attr, id_attr, tile_attr]
    multi_tile_lakes.crs_string = main_crs_str

    for l, lake in enumerate(multi_tile_lake_indices):
        lake_geom = can_vec.features[l].GetGeometryRef()
        lake_attr = can_vec.attributes[l]

        multi_tile_lakes.add_feat(geom=lake_geom,
                                  attr=lake_attr)

    print(multi_tile_lakes)
    multi_tile_lakes.write_vector(out_dir + 'multi_tile_lakes.shp')

    print('Step completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')
    sys.stdout.flush()

    print('Writing all lakes vector for Canada and Alaska...')
    can_vec.write_vector(outfile=out_file)

    print('Completed at {}'.format(Timer.display_time(time.time() - t)))
    print('----------------------------------------------------------------')

