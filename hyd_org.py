from osgeo import osr, ogr
from common import Vector, Timer, File
import json


@Timer.timing()
def reproj_and_find_ids(in_vec,
                        bounds_vec,
                        verbose=False):

    """
    Method to reproject the lakes vector and find intersecting tile ids
    :param in_vec: input lakes vector
    :param bounds_vec: input tile vector
    :return: reprojected lakes vector and tile id list
    """

    out_vec = Vector()
    out_vec.type = in_vec.type
    out_vec.nfeat = in_vec.nfeat

    destination_spatial_ref = osr.SpatialReference()
    res = destination_spatial_ref.ImportFromWkt(bounds_vec.crs_string)

    out_vec.spref = destination_spatial_ref
    out_vec.crs_string = destination_spatial_ref.ExportToWkt()

    # get source spatial reference from Spatial reference WKT string in self
    source_spatial_ref = in_vec.spref

    # create a transform tool (or driver)
    transform_tool = osr.CoordinateTransformation(source_spatial_ref,
                                                  destination_spatial_ref)

    # Create a memory layer
    memory_driver = ogr.GetDriverByName('Memory')
    out_vec.data_source = memory_driver.CreateDataSource('out')

    # create a layer in memory
    out_vec.layer = out_vec.data_source.CreateLayer('temp',
                                                    srs=source_spatial_ref,
                                                    geom_type=in_vec.type)

    # initialize new feature list
    out_vec.features = list()
    out_vec.fields = list()
    out_vec.name = in_vec.name

    # input layer definition
    in_layer_definition = in_vec.layer.GetLayerDefn()

    # add fields
    for i in range(0, in_layer_definition.GetFieldCount()):
        field_definition = in_layer_definition.GetFieldDefn(i)
        out_vec.layer.CreateField(field_definition)
        out_vec.fields.append(field_definition)

    # layer definition with new fields
    temp_layer_definition = out_vec.layer.GetLayerDefn()

    out_vec.wkt_list = list()
    out_vec.attributes = in_vec.attributes

    intersect_list = list()

    # convert each feature
    for j, feat in enumerate(in_vec.features):

        # transform geometry
        temp_geom = feat.GetGeometryRef()
        temp_geom.Transform(transform_tool)

        tile_intersects = list()

        for i, b_feat in enumerate(bounds_vec.features):

            if verbose:
                print('Checking tile {} of {} with geom {} of {}'.format(str(i+1),
                                                                         str(bounds_vec.nfeat),
                                                                         str(j+1),
                                                                         str(in_vec.nfeat)))

            b_geom = b_feat.GetGeometryRef()

            if temp_geom.Intersects(b_geom):
                tile_intersects.append(bounds_vec.attributes[i]['grid_id'])

        intersect_list.append(tile_intersects)

        out_vec.wkt_list.append(temp_geom.ExportToWkt())

        # create new feature using geometry
        temp_feature = ogr.Feature(temp_layer_definition)
        temp_feature.SetGeometry(temp_geom)

        # fill geometry fields
        for i in range(0, temp_layer_definition.GetFieldCount()):
            field_definition = temp_layer_definition.GetFieldDefn(i)
            temp_feature.SetField(field_definition.GetNameRef(), feat.GetField(i))

        # add the feature to the shapefile
        out_vec.layer.CreateFeature(temp_feature)

        out_vec.features.append(temp_feature)

    return out_vec, intersect_list


if __name__ == '__main__':

    # file paths
    ak_file = "C:/temp/dem/hydroFlat/vectors/US_AK/3_merged/NHD_AK_WB_noGlac_diss_gte1000m2.shp"
    can_file_dir = "C:/temp/dem/hydroFlat/vectors/CAN/2_shps/indiv/"
    tile_file = "C:/temp/dem/hydroFlat/grids/PCS_NAD83_C_grid_ABoVE_intersection.shp"
    out_file = "C:/temp/dem/hydroFlat/vectors/alaska_canada_lakes.shp"

    # read tiles in memory
    tile_vec = Vector(filename=tile_file, verbose=True)
    main_crs_str = tile_vec.crs_string

    # make border vector in memory
    border_json = {"type": "Polygon", "coordinates": [[[-142.277, 70.526], [-141.749, 60.199], [-130.675, 54.165],
                                                       [-128.126, 55.680], [-135.421, 60.111], [-137.355, 59.715],
                                                       [-139.904, 61.061], [-140.167, 70.438]]]}

    border_spref = osr.SpatialReference()
    border_spref.ImportFromEPSG(4326)

    border_json_str = json.dumps(border_json)
    border_geom = ogr.CreateGeometryFromJson(border_json_str)
    border_geom = Vector.reproj_geom(border_geom,
                                     border_spref.ExportToWkt(),
                                     main_crs_str)

    border_vec = Vector(in_memory=True,
                        spref_str=border_spref.ExportToWkt(),
                        geom_type=Vector.ogr_geom_type('polygon'))

    border_vec.name = 'border'
    border_vec.add_geom(border_geom)

    print(border_vec)

    can_file_obj = File()
    can_file_obj.dirpath = can_file_dir
    can_file_list = can_file_obj.find_all('*.shp')

    can_vec = Vector(in_memory=True,
                     spref_str=main_crs_str,
                     geom_type=Vector.ogr_geom_type('polygon'))

    can_vec.name = 'canada_lakes'
    filename_attr = ogr.FieldDefn('filename', ogr.OFTString)
    can_vec.layer.CreateField(filename_attr)
    id_attr = ogr.FieldDefn('orig_id', ogr.OFTString)
    can_vec.layer.CreateField(id_attr)
    can_vec.fields = can_vec.fields + [filename_attr, id_attr]

    temp_count = 0
    error_list = list()
    for i, can_file in enumerate(can_file_list):
        print('File {} of {}: {}'.format(str(i+1),
                                         str(len(can_file_list)),
                                         can_file))
        temp_vec = None

        try:
            temp_vec = Vector(filename=can_file,
                              spref_str=main_crs_str)

        except AttributeError:
            error_list.append('Error reading {}'.format(can_file))
            print('Error reading {}'.format(can_file))
        if temp_vec:
            for feat in temp_vec.features:
                geom = feat.GetGeometryRef()
                orig_id = feat.items()['NID']
                filename = File(can_file).basename.split('.shp')[0]

                can_vec.add_geom(geom,
                                 attr={'filename': filename,
                                       'orig_id': orig_id})

            temp_count = temp_count + temp_vec.nfeat

    print(can_vec)

    for error in error_list:
        print(error)

    ak_vec = Vector(ak_file, verbose=True)

    print(can_vec)
    print(ak_vec)
    print(tile_vec)

    print(ak_vec.crs_string)
    print(can_vec.crs_string)
    print(tile_vec.crs_string)

    exit()

    lakes_reproj, id_list = reproj_and_find_ids(ak_vec,
                                                tile_vec,
                                                verbose=True)

    multi_tile_lake_idx = list(i for i in range(0, len(id_list)) if len(id_list[i]) > 1)

    print(multi_tile_lake_idx)
    print(len(multi_tile_lake_idx))




