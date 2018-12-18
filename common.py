import os
import numpy as np
from osgeo import gdal, osr, gdal_array, ogr


class Raster(object):
    """
    Class for Raster .tif files
    """
    def __init__(self,
                 filename=None):
        """
        Constructor for class Raster
        :param filename: Full file name of the Raster file with file path (e.g.: /scratch/user/raster_file.tif)
        """
        if filename is not None and os.path.isfile(filename):

            self.filename = filename
            self.metadata = dict()

            fileptr = gdal.Open(self.filename)

            # update some metadata values
            self.metadata['nbands'] = fileptr.RasterCount
            self.metadata['nrows'] = fileptr.RasterYSize
            self.metadata['ncols'] = fileptr.RasterXSize

            # get transform
            self.metadata['transform'] = fileptr.GetGeoTransform()
            self.metadata['spref'] = fileptr.GetProjectionRef()

            # read raster as array
            self.array = np.zeros((self.metadata['nbands'],
                                   self.metadata['nrows'],
                                   self.metadata['ncols']),
                                  gdal_array.
                                  GDALTypeCodeToNumericTypeCode(fileptr.GetRasterBand(1).
                                                                DataType))

            self.metadata['bandname'] = list()

            # loop through all bands
            for i in range(0, self.metadata['nbands']):
                temp_band = fileptr.GetRasterBand(i+1)

                # get data type of first band
                if i == 0:
                    self.metadata['datatype'] = temp_band.DataType
                    self.metadata['nodatavalue'] = temp_band.GetNoDataValue()

                # write raster to array
                self.array[i, :, :] = temp_band.ReadAsArray()

                # read band names
                self.metadata['bandname'].append(temp_band.GetDescription())

            fileptr = None

            print('\nInitialized Raster {} of shape {}x{}x{}'.format(os.path.basename(self.filename),
                                                                     self.metadata['nbands'],
                                                                     self.metadata['nrows'],
                                                                     self.metadata['ncols']))
        else:

            self.filename = filename
            self.metadata = dict()
            self.array = np.empty(shape=[0, 0, 0])

            print('\nInitialized empty Raster')

    def __repr__(self):
        return "<Raster {} of shape {}x{}x{} at {}>".format(os.path.basename(self.filename),
                                                            self.metadata['nbands'],
                                                            self.metadata['nrows'],
                                                            self.metadata['ncols'],
                                                            hex(id(self)))

    def vector_extract(self,
                       vector,
                       pctl=25,
                       band_index=0,
                       min_pixels=25,
                       replace=False):

        """
        Method to extract percentile values from a raster based on vector bounds and
        replace the raster pixels inside vector boundary with the percentile value
        :param vector: Vector class object
        :param pctl: Percentile to be calculated
        :param band_index: Which band to operate on
        :param min_pixels: Number of minimum pixels for extraction (default: 25)
        :param replace: If the raster pixels should be replaced by the calculated percentile value
        :return: List of percentiles by vector features
        """

        extract_list = list()

        if replace:
            out_arr = np.copy(self.array)
        else:
            out_arr = None

        # loop through all vector features
        for i in range(0, vector.nfeat):

            # convert vector wkt string to geometry
            geom = ogr.CreateGeometryFromWkt(vector.wkt_list[i])
            spref = osr.SpatialReference()
            result = spref.ImportFromWkt(self.metadata['spref'])

            # get driver to write to memory
            memory_driver = ogr.GetDriverByName('Memory')
            temp_datasource = memory_driver.CreateDataSource('temp')

            # get geometry type (e.g. polygon)
            geom_type = geom.GetGeometryType()

            # create layer in memory
            temp_layer = temp_datasource.CreateLayer('temp_layer',
                                                     srs=spref,
                                                     geom_type=geom_type)

            # attributes
            fielddefn = ogr.FieldDefn('fid', ogr.OFTInteger)
            result = temp_layer.CreateField(fielddefn)

            res = temp_layer.CreateField(fielddefn)
            layerdef = temp_layer.GetLayerDefn()

            # create feature in layer
            temp_feature = ogr.Feature(layerdef)
            temp_feature.SetGeometry(geom)
            temp_feature.SetField('fid', 0)
            temp_layer.CreateFeature(temp_feature)

            # create blank raster in memory
            ras_driver = gdal.GetDriverByName('MEM')
            ras_ds = ras_driver.Create('',
                                       self.metadata['ncols'],
                                       self.metadata['nrows'],
                                       self.metadata['nbands'],
                                       gdal.GDT_Byte)

            # set geolocation parameters for raster
            ras_ds.SetProjection(self.metadata['spref'])
            ras_ds.SetGeoTransform(self.metadata['transform'])

            # rasterize the geometry and burn to raster to create a mask
            result = gdal.RasterizeLayer(ras_ds,
                                         [1],
                                         temp_layer,
                                         None,
                                         None,
                                         [1],
                                         ['ALL_TOUCHED=TRUE'])

            # read mask band as array
            temp_band = ras_ds.GetRasterBand(1)
            mask_arr = temp_band.ReadAsArray()

            # make list of unmasked pixels
            pixel_xy_loc = [(y, x) for y, x in np.transpose(np.where(mask_arr == 1))]

            # calculate the percentile value
            pixel_xyz_loc = list((band_index,) + ptup for ptup in pixel_xy_loc)
            temp_vals = list(self.array.item(loc) for loc in pixel_xyz_loc)

            # extract pixels other than the no-data value
            pixel_vals = list(val for val in temp_vals if val != self.metadata['nodatavalue'])

            # replace only if the number of pixels is greater than min value
            if len(pixel_vals) < min_pixels:
                pctl_val = None

            else:
                # calculate percentile value
                pctl_val = np.percentile(pixel_vals, pctl)

                # if replaced specified, replace pixels in the raster array
                if replace:
                    for loc in pixel_xyz_loc:
                        out_arr[loc] = pctl_val

                    self.array = out_arr

            extract_list.append(pctl_val)

        temp_layer = ras_ds = temp_datasource = None

        return extract_list

    def get_bounds(self):
        """
        Method to extract bounding rectangle for a raster as a Vector object
        :return: Vector object
        """

        mcx = self.metadata['transform'][0]
        mcy = self.metadata['transform'][3]

        px = np.abs(self.metadata['transform'][1])
        py = np.abs(self.metadata['transform'][5])

        nl = self.metadata['nrows']
        ns = self.metadata['ncols']

        bounds_list = [[mcx, mcy],
                       [mcx + ns * px, mcy],
                       [mcx + ns * px, mcy - nl * py],
                       [mcx, mcy - nl * py],
                       [mcx, mcy]]

        bounds_wkt = "POLYGON(({}))".format(', '.join(list(' '.join([str(x), str(y)]) for (x, y) in bounds_list)))

        bounds_vector = Vector()

        bounds_vector.wkt_list.append(bounds_wkt)
        bounds_vector.crs_string = self.metadata['spref']
        bounds_vector.nfeat = 1
        bounds_vector.type = 3
        bounds_vector.name = os.path.basename(self.filename).split('.')[0] + '_bounds'
        bounds_vector.spref = osr.SpatialReference()
        res = bounds_vector.spref.ImportFromWkt(bounds_vector.crs_string)

        memory_driver = ogr.GetDriverByName('Memory')
        bounds_vector.data_source = memory_driver.CreateDataSource('out')
        bounds_vector.layer = bounds_vector.data_source.CreateLayer('image_bounds',
                                                                    srs=bounds_vector.spref,
                                                                    geom_type=bounds_vector.type)

        fielddefn = ogr.FieldDefn('fid', ogr.OFTInteger)
        bounds_vector.fields.append(fielddefn)

        res = bounds_vector.layer.CreateField(fielddefn)

        layer_def = bounds_vector.layer.GetLayerDefn()
        geom = ogr.CreateGeometryFromWkt(bounds_wkt)
        feat = ogr.Feature(layer_def)
        feat.SetGeometry(geom)
        feat.SetField('fid', 0)

        bounds_vector.layer.CreateFeature(feat)
        bounds_vector.features = [feat]

        return bounds_vector

    def write_raster(self,
                     outfile,
                     file_type='GTiff'):
        """
        Method to write raster to disk
        :param outfile: Output file name
        :param file_type: File type (default: 'GTiff')
        :return: None
        """
        print('\nWriting {} raster: {}'.format(file_type,
                                               outfile))

        if self.array is not None and self.metadata['datatype'] is None:
            self.metadata['datatype'] = gdal_array.NumericTypeCodeToGDALTypeCode(self.array.dtype.type)

        # initiate output
        driver = gdal.GetDriverByName(file_type)
        ptr = driver.Create(outfile,
                            self.metadata['ncols'],
                            self.metadata['nrows'],
                            self.metadata['nbands'],
                            self.metadata['datatype'])

        # set projection parameters
        ptr.SetGeoTransform(self.metadata['transform'])
        ptr.SetProjection(self.metadata['spref'])

        # loop thru band array
        for i in range(0, self.metadata['nbands']):
            band = self.array[i, :, :]

            if self.metadata['bandname'] is not None:
                if isinstance(self.metadata['bandname'], list):
                    bandname = str(self.metadata['bandname'][i])
                else:
                    bandname = str(self.metadata['bandname'])
            else:
                bandname = 'Band_{}'.format(str(i + 1))

            ptr.GetRasterBand(i + 1).WriteArray(band)
            ptr.GetRasterBand(i + 1).SetDescription(bandname)
            if self.metadata['nodatavalue'] is not None:
                ptr.GetRasterBand(i + 1).SetNoDataValue(self.metadata['nodatavalue'])

        # delete pointers
        ptr.FlushCache()  # save to disk
        ptr = None


class Vector(object):
    """
    Class for vector objects
    """
    def __init__(self,
                 filename=None,
                 layer_index=0):
        """
        Constructor for class Vector
        :param filename: Name of the vector file (shapefile) with full path
        :param layer_index: Index of the vector layer to pull (default: 0)
        """

        self.filename = filename
        self.nfeat = None
        self.features = list()
        self.attributes = list()
        self.wkt_list = list()
        self.data_source = None
        self.layer = None
        self.spref = None
        self.crs_string = None
        self.type = None
        self.name = 'Empty'
        self.nfeat = None
        self.fields = list()

        if filename is not None and os.path.isfile(filename):

            # open vector file
            self.data_source = ogr.Open(self.filename)

            # get layer
            self.layer = self.data_source.GetLayerByIndex(layer_index)

            # spatial reference
            self.spref = self.layer.GetSpatialRef()
            self.crs_string = self.spref.ExportToWkt()

            # other layer metadata
            self.type = self.layer.GetGeomType()
            self.name = self.layer.GetName()

            # extract feature attributes
            # and feature geometry feature string

            # get field defintions
            layer_definition = self.layer.GetLayerDefn()
            self.fields = [layer_definition.GetFieldDefn(i) for i in range(0, layer_definition.GetFieldCount())]

            # iterate thru features and append to list
            feat = self.layer.GetNextFeature()

            while feat:
                all_items = feat.items()
                geom = feat.geometry()

                self.attributes.append(all_items)
                self.features.append(feat)
                self.wkt_list.append(geom.ExportToWkt())

                feat = self.layer.GetNextFeature()

            self.nfeat = self.layer.GetFeatureCount()

            print("\nInitialized Vector {} of type {} ".format(self.name,
                                                               self.type) +
                  "with {} feature(s) and {} attribute(s)".format(str(self.nfeat),
                                                                  str(len(self.fields))))

        else:
            print("\nInitialized empty Vector")

    def __repr__(self):
        return "<Vector {} of type {} ".format(self.name,
                                               self.type) + \
               "with {} feature(s) and {} attribute(s) >".format(str(self.nfeat),
                                                                 str(len(self.fields)))

    def get_intersecting_vector(self,
                                query_vector,
                                index=False):
        """
        Gets tiles intersecting with the given geometry (any type).
        This method returns an initialized Vector object. The first argument (or self) should be Polygon type.
        :param query_vector: Initialized vector object to query with (geometry could be any type)
        :param index: If the index of self vector where intersecting, should be returned
        :returns: Vector object of polygon features from self
        """

        query_list = list()

        # determine if same coordinate system
        if self.spref.IsSame(query_vector.spref) == 1:

            index_list = list()

            # determine which features intersect
            for j in range(0, query_vector.nfeat):
                qgeom = query_vector.features[j].GetGeometryRef()

                for i in range(0, self.nfeat):

                    feat = self.features[i]
                    geom = feat.GetGeometryRef()

                    if geom.Intersects(qgeom):
                        index_list.append(i)

            intersect_index = sorted(set(index_list))

            for feat_index in intersect_index:
                feat = self.features[feat_index]

                temp_feature = dict()
                temp_feature['feat'] = feat
                temp_feature['attr'] = feat.items()

                query_list.append(temp_feature)

            # create output vector in memory
            out_vector = Vector()

            # create a vector in memory
            memory_driver = ogr.GetDriverByName('Memory')
            temp_datasource = memory_driver.CreateDataSource('out_vector')

            # relate memory vector source to Vector object
            out_vector.mem_source = temp_datasource
            out_vector.data_source = temp_datasource
            out_vector.wkt_list = list()

            # update features and crs
            out_vector.nfeat = len(query_list)
            out_vector.type = self.type
            out_vector.spref = self.spref
            out_vector.fields = self.fields
            out_vector.name = self.name

            # create layer in memory
            temp_layer = temp_datasource.CreateLayer('temp_layer',
                                                     srs=self.spref,
                                                     geom_type=self.type)

            # create the same attributes in the temp layer as the input Vector layer
            for k in range(0, len(self.fields)):
                temp_layer.CreateField(self.fields[k])

            # fill the features in output layer
            for i in range(0, len(query_list)):

                # create new feature
                temp_feature = ogr.Feature(temp_layer.GetLayerDefn())

                # fill geometry
                temp_geom = query_list[i]['feat'].GetGeometryRef()
                temp_feature.SetGeometry(temp_geom)

                # get attribute dictionary from query list
                attr_dict = dict(query_list[i]['attr'].items())

                # set attributes for the feature
                for j in range(0, len(self.fields)):
                    name = self.fields[j].GetName()
                    temp_feature.SetField(name, attr_dict[name])

                out_vector.features.append(temp_feature)
                out_vector.wkt_list.append(temp_geom.ExportToWkt())

                # create new feature in output layer
                temp_layer.CreateFeature(temp_feature)

            out_vector.layer = temp_layer

            if index:
                return out_vector, intersect_index
            else:
                return out_vector

        else:
            raise RuntimeError("Coordinate system or object type mismatch")

    def reproject(self,
                  epsg=None,
                  dest_spatial_ref_str=None,
                  dest_spatial_ref_str_type=None,
                  destination_spatial_ref=None,
                  _return=False):
        """
        Transfrom a geometry using OSR library (which is based on PROJ4)
        :param dest_spatial_ref_str: Destination spatial reference string
        :param dest_spatial_ref_str_type: Destination spatial reference string type
        :param destination_spatial_ref: OSR spatial reference object for destination feature
        :param epsg: Destination EPSG SRID code (default: 4326)
        :return: Reprojected vector object
        Todo: add write to file method
        """

        vector = Vector()
        vector.type = self.type
        vector.nfeat = self.nfeat

        if destination_spatial_ref is None:
            destination_spatial_ref = osr.SpatialReference()

            if dest_spatial_ref_str is not None:
                if dest_spatial_ref_str_type == 'wkt':
                    res = destination_spatial_ref.ImportFromWkt(dest_spatial_ref_str)
                elif dest_spatial_ref_str_type == 'proj4':
                    res = destination_spatial_ref.ImportFromProj4(dest_spatial_ref_str)
                elif dest_spatial_ref_str_type == 'epsg':
                    res = destination_spatial_ref.ImportFromEPSG(dest_spatial_ref_str)
                else:
                    raise ValueError("No spatial reference string type specified")
            elif epsg is not None:
                res = destination_spatial_ref.ImportFromEPSG(epsg)

            else:
                raise ValueError("Destination spatial reference not specified")

        vector.spref = destination_spatial_ref
        vector.crs_string = destination_spatial_ref.ExportToWkt()

        # get source spatial reference from Spatial reference WKT string in self
        source_spatial_ref = self.spref

        # create a transform tool (or driver)
        transform_tool = osr.CoordinateTransformation(source_spatial_ref,
                                                      destination_spatial_ref)

        # Create a memory layer
        memory_driver = ogr.GetDriverByName('Memory')
        vector.data_source = memory_driver.CreateDataSource('out')

        # create a layer in memory
        vector.layer = vector.data_source.CreateLayer('temp',
                                                      srs=source_spatial_ref,
                                                      geom_type=self.type)

        # initialize new feature list
        vector.features = list()
        vector.fields = list()
        vector.name = self.name

        # input layer definition
        in_layer_definition = self.layer.GetLayerDefn()

        # add fields
        for i in range(0, in_layer_definition.GetFieldCount()):
            field_definition = in_layer_definition.GetFieldDefn(i)
            vector.layer.CreateField(field_definition)
            vector.fields.append(field_definition)

        # layer definition with new fields
        temp_layer_definition = vector.layer.GetLayerDefn()

        vector.wkt_list = list()
        vector.attributes = self.attributes

        # convert each feature
        for feat in self.features:

            # transform geometry
            temp_geom = feat.GetGeometryRef()
            temp_geom.Transform(transform_tool)

            vector.wkt_list.append(temp_geom.ExportToWkt())

            # create new feature using geometry
            temp_feature = ogr.Feature(temp_layer_definition)
            temp_feature.SetGeometry(temp_geom)

            # fill geometry fields
            for i in range(0, temp_layer_definition.GetFieldCount()):
                field_definition = temp_layer_definition.GetFieldDefn(i)
                temp_feature.SetField(field_definition.GetNameRef(), feat.GetField(i))

            # add the feature to the shapefile
            vector.layer.CreateFeature(temp_feature)

            vector.features.append(temp_feature)
            vector.epsg = epsg

        if _return:
            return vector
        else:
            self.layer = vector.layer
            self.features = vector.features
            self.fields = vector.fields
            self.data_source = vector.data_source
            self.wkt_list = vector.wkt_list
            self.crs_string = vector.crs_string


def _find_dict_index(list_of_dicts,
                     key,
                     value):
    """
    Method to find where a key value pair occurs in a list of dictionaries
    :param list_of_dicts: List of dictionaries
    :param key: Key to find
    :param value: Value associated with the key
    :return: index (int)
    """
    for i, i_dict in enumerate(list_of_dicts):
        for k, v in i_dict.items():
            if k == key:
                if v == value:
                    return i
