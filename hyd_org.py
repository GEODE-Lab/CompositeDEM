import os
import sys
import numpy as np
from osgeo import gdal, osr, gdal_array, ogr
from common import Vector, Timer


@Timer.timing()
def reproj_and_find_ids(in_vec,
                        bounds_vec):

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
    for feat in in_vec.features:

        # transform geometry
        temp_geom = feat.GetGeometryRef()
        temp_geom.Transform(transform_tool)

        tile_intersects = list()

        for i, b_feat in enumerate(bounds_vec.features):
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

    ak_file = "f:/hydroFlat/vectors/US_AK/3_merged/NHD_AK_WB_noGlac_diss_gte1000m2.shp"
    can_file_dir = "f:/hydroFlat/vectors/CAN/2_shps/indiv/"
    tile_file = "f:/hydroFlat/grids/PCS_NAD83_C_grid_ABoVE_intersection.shp"

    ak_vec = Vector(ak_file, verbose=True)

    tile_vec = Vector(tile_file, verbose=True)

    print(ak_vec)
    print(tile_vec)

    print(ak_vec.crs_string)
    print(tile_vec.crs_string)

    lakes_reproj, id_list = reproj_and_find_ids(ak_vec, tile_vec)

    multi_tile_lake_idx = list(i for i in range(0, len(id_list)) if len(id_list[i]) > 1)

    print(multi_tile_lake_idx)
    print(len(multi_tile_lake_idx))




