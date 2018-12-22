import os
import json
import time
import shutil
import fnmatch
import numpy as np
from functools import wraps
from osgeo import gdal, osr, gdal_array, ogr
gdal.UseExceptions()
ogr.UseExceptions()


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
                 layer_index=0,
                 spref_str=None,
                 geom_type=3,
                 in_memory=False,
                 verbose=False):
        """
        Constructor for class Vector
        :param filename: Name of the vector file (shapefile) with full path
        :param layer_index: Index of the vector layer to pull (default: 0)
        """

        self.filename = filename
        self.features = list()
        self.attributes = list()
        self.wkt_list = list()
        self.data_source = None
        self.layer = None
        self.spref = None
        self.crs_string = None
        self.type = None
        self.name = 'Empty'
        self.nfeat = 0
        self.fields = list()
        self.data = list()

        if filename is not None and os.path.isfile(filename):

            # open vector file
            self.data_source = ogr.Open(self.filename)
            file_layer = self.data_source.GetLayerByIndex(layer_index)

            if in_memory:
                out_driver = ogr.GetDriverByName('Memory')
                out_datasource = out_driver.CreateDataSource('mem_source')
                self.layer = out_datasource.CopyLayer(file_layer, 'mem_source')
                self.data_source = out_datasource
                file_layer = None

            else:
                # get layer
                self.layer = file_layer

            # spatial reference
            self.spref = self.layer.GetSpatialRef()

            if spref_str is not None:
                dest_spref = osr.SpatialReference()
                res = dest_spref.ImportFromWkt(spref_str)

                if self.spref.IsSame(dest_spref) == 1:
                    dest_spref = None
            else:
                dest_spref = None

            self.crs_string = self.spref.ExportToWkt()

            # other layer metadata
            self.type = self.layer.GetGeomType()
            self.name = self.layer.GetName()

            if verbose:
                print('Reading vector {} of type {} ...'.format(self.name,
                                                                str(self.type)))

            # get field defintions
            layer_definition = self.layer.GetLayerDefn()
            self.fields = [layer_definition.GetFieldDefn(i) for i in range(0, layer_definition.GetFieldCount())]

            # number of features
            self.nfeat = self.layer.GetFeatureCount()

            # if the vector should be initialized in some other spatial reference
            if dest_spref is not None:
                transform_tool = osr.CoordinateTransformation(self.spref,
                                                              dest_spref)
                self.spref = dest_spref
            else:
                transform_tool = None

            # iterate thru features and append to list
            feat = self.layer.GetNextFeature()

            feat_count = 0
            while feat:
                # extract feature attributes
                all_items = feat.items()

                # and feature geometry feature string
                geom = feat.GetGeometryRef()

                if self.type == 3:
                    geom.CloseRings()

                # convert to another projection and write new features
                if dest_spref is not None:
                    geom.Transform(transform_tool)

                    new_feat = ogr.Feature(layer_definition)
                    for attr, val in all_items.items():
                        new_feat.SetField(attr, val)
                    new_feat.SetGeometry(geom)
                else:
                    new_feat = feat

                if verbose:
                    attr_dict = json.dumps(all_items)
                    print('Feature {} of {} : attr {}'.format(str(feat_count+1),
                                                              str(self.nfeat),
                                                              attr_dict))

                self.attributes.append(all_items)
                self.features.append(new_feat)
                self.wkt_list.append(geom.ExportToWkt())
                feat_count += 1

                feat = self.layer.GetNextFeature()

            if verbose:
                print("\nInitialized Vector {} of type {} ".format(self.name,
                                                                   self.ogr_geom_type(self.type)) +
                      "with {} feature(s) and {} attribute(s)".format(str(self.nfeat),
                                                                      str(len(self.fields))))

        else:
            if in_memory:
                out_driver = ogr.GetDriverByName('Memory')
                out_datasource = out_driver.CreateDataSource('mem_source')
                self.data_source = out_datasource
                self.type = geom_type

                self.spref = osr.SpatialReference()
                res = self.spref.ImportFromWkt(spref_str)

                self.layer = self.data_source.CreateLayer('mem_layer',
                                                          srs=self.spref,
                                                          geom_type=geom_type)
                fid = ogr.FieldDefn('fid', ogr.OFTInteger)
                self.layer.CreateField(fid)
                self.fields = [fid]

            if verbose:
                print("\nInitialized empty Vector")

    def __repr__(self):
        return "<Vector {} of type {} ".format(self.name,
                                               self.ogr_geom_type(self.type)) + \
               "with {} feature(s) and {} attribute(s) >".format(str(self.nfeat),
                                                                 str(len(self.fields)))

    @staticmethod
    def ogr_data_type(x):
        """
        Method to get OGR data type, for use in creating OGR geometry fields
        :param x: Any data input
        :return: OGR data type
        """
        val = type(x).__name__.lower()

        val_dict = {
            'int': ogr.OFTInteger,
            'long': ogr.OFTInteger,
            'float': ogr.OFTReal,
            'double': ogr.OFTReal,
            'str': ogr.OFTString,
            'bool': ogr.OFTInteger,
            'nonetype': ogr.OFSTNone,
            'none': ogr.OFSTNone,
        }

        try:
            return val_dict[val]
        except (KeyError, NameError):
            return val_dict['nonetype']

    @staticmethod
    def ogr_geom_type(x):
        """
        Method to return OGR geometry type from input string
        :param x: String to convert to OGR geometry type code
        :return: OGR geometry type code
        """

        if type(x).__name__ == 'str':
            comp_str = x.lower()
            comp_dict = {
                'point': 1,
                'line': 2,
                'linestring': 2,
                'polygon': 3,
                'multipoint': 4,
                'multilinestring': 5,
                'multipolygon': 6,
                'geometry': 0,
                'no geometry': 100
            }
            try:
                return comp_dict[comp_str]
            except (KeyError, NameError):
                return None

        elif type(x).__name__ == 'int' or type(x).__name__ == 'float':
            comp_dict = {
                1: 'point',
                2: 'linestring',
                3: 'polygon',
                4: 'multipoint',
                5: 'multilinestring',
                6: 'multipolygon',
                0: 'geometry',
                100: 'no geometry',
            }
            try:
                return comp_dict[int(x)].upper()
            except (KeyError, NameError):
                return None

        else:
            raise(ValueError('Invalid format'))

    @staticmethod
    def string_to_ogr_type(x):
        """
        Method to return name of the data type
        :param x: input item
        :return: string
        """
        if type(x).__name__ != 'str':
            return Vector.ogr_data_type(x)
        else:
            try:
                val = int(x)
            except ValueError:
                try:
                    val = float(x)
                except ValueError:
                    try:
                        val = bool(x)
                    except ValueError:
                        try:
                            val = str(x)
                        except:
                            val = None

            return Vector.ogr_data_type(val)

    def add_feat(self,
                 geom,
                 primary_key='fid',
                 attr=None):

        """
        Add geometry as a feature to a Vector in memory
        :param geom: osgeo geometry
        :param primary_key: primary key for the attribute table
        :param attr: Attributes
        :return: None
        """

        feat = ogr.Feature(self.layer.GetLayerDefn())
        feat.SetGeometry(geom)

        if attr is not None:
            for k, v in attr.items():
                feat.SetField(k, v)
            if primary_key is not None:
                if primary_key not in attr:
                    feat.SetField(primary_key, self.nfeat)
        else:
            if primary_key is not None:
                feat.SetField(primary_key, self.nfeat)

        self.layer.CreateFeature(feat)
        self.features.append(feat)
        self.wkt_list.append(geom.ExportToWkt())
        if attr is not None:
            self.attributes.append(attr)
        elif primary_key is not None:
            self.attributes.append({primary_key, self.nfeat})

        self.nfeat += 1

    def merge(self,
              vector,
              remove=False):

        """
        Method to merge two alike vectors. This method only works for vectors
        that have same spref or crs_string, attribute keys, and geom types
        :param vector: Vector to merge in self
        :param remove: if the vector should be removed after merging
        :return: None
        """

        for i, feat in enumerate(vector.features):
            geom = feat.GetGeometryRef()
            attr = feat.items()

            self.add_feat(geom=geom,
                          attr=attr)

            if len(vector.data) > 0:
                self.data.append(vector.data[i])

        if remove:
            vector = None

    def write_vector(self,
                     outfile=None,
                     in_memory=False):
        """
        Method to write the vector object to memory or to file
        :param outfile: File to write the vector object to
        :param in_memory: If the vector object should be written in memory (default: False)
        :return: Vector object if written to memory else NoneType
        """

        if in_memory:

            driver_type = 'Memory'

            if outfile is not None:
                outfile = os.path.basename(outfile).split('.')[0]
            else:
                outfile = 'in_memory'

            out_driver = ogr.GetDriverByName(driver_type)
            out_datasource = out_driver.CreateDataSource(outfile)
            out_layer = out_datasource.CopyLayer(self.layer, outfile)

            out_vector = Vector()

            out_vector.data_source = out_datasource
            out_vector.mem_source = out_datasource

            return out_vector

        else:

            if outfile is None:
                outfile = self.filename
                if self.filename is None:
                    raise ValueError("No filename for output")

            if os.path.basename(outfile).split('.')[-1] == 'json':
                driver_type = 'GeoJSON'
            elif os.path.basename(outfile).split('.')[-1] == 'csv':
                driver_type = 'Comma Separated Value'
            else:
                driver_type = 'ESRI Shapefile'

            out_driver = ogr.GetDriverByName(driver_type)
            out_datasource = out_driver.CreateDataSource(outfile)

            out_layer = out_datasource.CreateLayer(os.path.basename(outfile).split('.')[0],
                                                   srs=self.crs_string,
                                                   geom_type=self.type)

            for attr_key, attr_val in self.attributes[0].items():
                    field = ogr.FieldDefn(attr_key, self.string_to_ogr_type(attr_val))
                    res = out_layer.CreateField(field)

            layer_defn = out_layer.GetLayerDefn()

            if len(self.wkt_list) > 0:
                for i, wkt_geom in enumerate(self.wkt_list):
                    geom = ogr.CreateGeometryFromWkt(wkt_geom)
                    feat = ogr.Feature(layer_defn)
                    feat.SetGeometry(geom)

                    for attr, val in self.attributes[i].items():
                        feat.SetField(attr, val)

                    out_layer.CreateFeature(feat)

            else:
                for feature in self.features:
                    out_layer.CreateFeature(feature)

            out_datasource = out_driver = None

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
        :param epsg: Destination EPSG SRID code
        :return: Reprojected vector object
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

    @staticmethod
    def reproj_geom(geoms,
                    source_spref_str,
                    dest_spref_str):

        """
        Method to reproject geometries
        :param geoms: List of osgeo geometries or a single geometry
        :param source_spref_str: Source spatial reference string
        :param dest_spref_str: Destination spatial reference string
        :return: osgeo geometry
        """

        source_spref = osr.SpatialReference()
        dest_spref = osr.SpatialReference()

        res = source_spref.ImportFromWkt(source_spref_str)
        res = dest_spref.ImportFromWkt(dest_spref_str)
        transform_tool = osr.CoordinateTransformation(source_spref,
                                                      dest_spref)

        if type(geoms).__name__ == 'list':
            for geom in geoms:
                geom.Transfrom(transform_tool)
        else:
            geoms.Transform(transform_tool)
        return geoms


class File(object):
    """
    Class to handle file and folder operations
    """
    def __init__(self,
                 filename=None):
        """
        Initialize the File class
        :param filename: Name of the file
        """
        self.sep = os.path.sep

        self.filename = filename

        if self.filename is not None:
            self.basename = os.path.basename(filename)
            self.dirpath = os.path.dirname(filename)
        else:
            self.basename = None
            self.dirpath = None

    def __repr__(self):
        """
        Object Representation
        :return: String
        """
        return '<File handler for {}>'.format(self.filename)

    def dir_create(self,
                   _return=False):
        """
        Create dir if it doesnt exist
        :param: directory to be created
        """
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        else:
            pass
        if _return:
            return self.dirpath

    def file_exists(self):
        """
        Check file existence
        :return: Bool
        """
        if self.filename is None:
            return
        else:
            return os.path.isfile(self.filename)

    def dir_exists(self):
        """
        Check folder existence
        :return: Bool
        """
        return os.path.isdir(self.dirpath)

    def dir_delete(self):
        """
        Delete a directory and everything in it
        :return:
        """
        shutil.rmtree(self.dirpath,
                      ignore_errors=True)

    def file_delete(self):
        """
        Delete a file
        """
        os.remove(self.filename)

    def file_copy(self,
                  destination_dir=None,
                  destination_file=None,):

        if destination_dir is not None:
            shutil.copyfile(self.filename,
                            destination_dir + self.sep + self.basename)
        elif destination_file is not None:
            shutil.copyfile(self.filename,
                            destination_file)

    def get_dir(self):
        """
        Get current dir name
        :return: string
        """
        return os.getcwd()

    def add_to_filename(self,
                        string,
                        remove_check=True,
                        remove_ext=False):
        """
        Add a string before file extension
        :param string: String to be added
        :param remove_check: Check file for removal (Default: True)
        :param remove_ext: Should the file extension be removed (Default: False)
        :return: file name
        """
        components = self.basename.split('.')

        if not remove_ext:

            if len(components) >= 2:
                outfile = self.dirpath + self.sep + '.'.join(components[0:-1]) + \
                       string + '.' + components[-1]
            else:
                outfile = self.basename + self.sep + components[0] + string

        else:

            if len(components) >= 2:
                outfile = self.dirpath + self.sep + '.'.join(components[0:-1]) + \
                       string
            else:
                outfile = self.basename + self.sep + components[0] + string

        if remove_check:
            File(outfile).file_remove_check()

        return outfile

    def file_remove_check(self):
        """
        Remove a file silently; if not able to remove, change the filename and move on
        :return filename
        """

        # if file does not exist, try to create dir
        if not os.path.isfile(self.filename):
            self.dir_create()

        # if file exists then try to delete or
        # get a filename that does not exist at that location
        counter = 1
        while os.path.isfile(self.filename):
            # print('File exists: ' + filename)
            # print('Deleting file...')
            try:
                os.remove(self.filename)
            except OSError:
                print('File already exists. Error deleting file!')
                components = self.basename.split('.')
                if len(components) < 2:
                    self.filename = self.dirpath + self.sep + self.basename + '_' + str(counter)
                else:
                    self.filename = self.dirpath + self.sep + ''.join(components[0:-1]) + \
                               '_(' + str(counter) + ').' + components[-1]
                # print('Unable to delete, using: ' + filename)
                counter = counter + 1
        return self.filename

    def find_all(self,
                 pattern='*'):
        """
        Find all the names that match pattern
        :param pattern: pattern to look for in the folder
        """
        result = []
        # search for a given pattern in a folder path
        if pattern == '*':
            search_str = '*'
        else:
            search_str = '*' + pattern + '*'

        for root, dirs, files in os.walk(self.dirpath):
            for name in files:
                if fnmatch.fnmatch(name, search_str):
                    result.append(os.path.join(root, name))

        return result  # list

    def find_multiple(self,
                      pattern_list):
        """
        Find all the names that match pattern
        :param pattern_list: List of patterns to look for in the folder
        """
        result = list()

        for i in range(0, len(pattern_list)):
            temp = self.find_all(pattern_list[i])

            for j in range(0, len(temp)):
                result.append(temp[j])

        return result


class Timer:
    """
    Decorator class to measure time a function takes to execute
    """
    def __init__(self,
                 func):
        self.func = func

    @staticmethod
    def display_time(seconds,
                     precision=3):
        """
        method to display time in human readable format
        :param seconds: Number of seconds
        :param precision: Decimal precision
        :return: String
        """

        # define denominations
        intervals = [('weeks', 604800),
                     ('days', 86400),
                     ('hours', 3600),
                     ('minutes', 60),
                     ('seconds', 1)]

        # initialize list
        result = list()

        # coerce to float
        dtype = type(seconds).__name__
        if dtype != 'int' or dtype != 'long' or dtype != 'float':
            try:
                seconds = float(seconds)
            except (TypeError, ValueError, NameError):
                print("Type not coercible to Float")

        # break denominations
        for name, count in intervals:
            if name != 'seconds':
                value = seconds // count
                if value:
                    seconds -= value * count
                    if value == 1:
                        name = name.rstrip('s')
                    value = str(int(value))
                    result.append("{v} {n}".format(v=value,
                                                   n=name))
            else:
                value = "{:.{p}f}".format(seconds,
                                          p=precision)
                result.append("{v} {n}".format(v=value,
                                               n=name))

        # join output
        return ' '.join(result)

    @classmethod
    def timing(cls):
        """
        Function to compute timing for input function
        :return: Function and prints time taken
        """

        def time_it(func):

            @wraps(func)
            def wrapper(*args, **kwargs):

                t1 = time.time()
                val = func(*args, **kwargs)
                t2 = time.time()

                # time to run
                t = Timer.display_time(t2 - t1)

                print("Time it took to run {}: {}\n".format(func.__name__,
                                                            t))
                return val

            return wrapper

        return time_it

