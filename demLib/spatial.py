import os
import sys
import json
import numpy as np
from osgeo import gdal, osr, ogr, gdal_array, gdalconst
gdal.UseExceptions()
ogr.UseExceptions()


__all__ = ['Raster',
           'Vector']


class Raster(object):
    """
    Class for Raster .tif files
    """
    def __init__(self,
                 filename=None,
                 datasource=None,
                 get_array=False,
                 offsets=None,
                 band_order=None):
        """
        Constructor for class Raster
        :param filename: Full file path of the Raster file (e.g.: /home/user/raster_file.tif)
        :param datasource: GDAL dataset shadow object
        :param get_array: If the Raster array should be extracted from file
        :param offsets: array offsets in x, y coordinates (x_start, y_start, n_columns, n_rows)
                        here,
                            x_start = starting pixel x location
                            y_start = starting pixel y location
                            n_columns = number of columns to extract
                            n_rows = number of rows to extract
                        default: entire file extent
        :param: Order of bands to be extracted. Bands can be repeated.
                example: [0,1,2,3,...], default: all bands

        Extracted array is 3 dimensional even if there is only one band: bands x rows x columns
        """
        if not hasattr(self, 'metadata'):
            self.metadata = {'nbands': None,
                             'nrows': None,
                             'ncols': None,
                             'bandnames': [],
                             'datatype': None,
                             'nodatavalue': None,
                             'transform': None,
                             'spref': None}

        if datasource is not None:
            self.datasource = datasource
            self.filename = filename
        elif filename is not None and os.path.isfile(filename):
            self.datasource = gdal.Open(filename)
            self.filename = filename
        else:
            self.filename = filename
            self.datasource = datasource
            return

        if band_order is None:
            self.metadata['nbands'] = self.datasource.RasterCount
            band_order = list(range(self.metadata['nbands']))
        else:
            self.metadata['nbands'] = len(band_order)

        self.metadata['transform'] = self.datasource.GetGeoTransform()

        if self.metadata['spref'] is None:
            self.metadata['spref'] = self.datasource.GetProjectionRef()

        if offsets is not None:
            self.metadata['nrows'] = offsets[3]
            self.metadata['ncols'] = offsets[2]
            self.array_offsets = offsets
            self.metadata['transform'][0] = self.metadata['transform'][0] + offsets[0] * self.metadata['transform'][1]
            self.metadata['transform'][3] = self.metadata['transform'][3] + offsets[1] * self.metadata['transform'][5]

        else:
            self.metadata['nrows'] = self.datasource.RasterYSize
            self.metadata['ncols'] = self.datasource.RasterXSize
            self.array_offsets = (0, 0, self.metadata['ncols'], self.metadata['nrows'])

        if get_array:
            self.metadata['bandnames'] = []
            self.array = np.zeros((len(band_order),
                                   self.array_offsets[3],
                                   self.array_offsets[2]),
                                  gdal_array.
                                  GDALTypeCodeToNumericTypeCode(self.datasource.GetRasterBand(1).
                                                                DataType))
        else:
            self.array = None

        # loop through all bands
        for ib in band_order:
            temp_band = self.datasource.GetRasterBand(ib+1)

            # get data type of first band
            if ib == 0:
                if self.metadata['datatype'] is None:
                    self.metadata['datatype'] = temp_band.DataType
                if self.metadata['nodatavalue'] is None:
                    self.metadata['nodatavalue'] = temp_band.GetNoDataValue()

            if get_array:
                self.array[ib, :, :] = temp_band.ReadAsArray(*self.array_offsets,
                                                             resample_alg=gdalconst.GRA_NearestNeighbour)
            # read band names
            self.metadata['bandnames'].append(temp_band.GetDescription())

    def __repr__(self):
        """
        Method to return a string representation of the Raster object if any Metadata value exists
        """
        if any(list(val for _, val in self.metadata.items())):
            return "<Raster {} of shape {}x{}x{} at {}>".format(os.path.basename(self.filename),
                                                                str(self.metadata['nbands']),
                                                                str(self.metadata['nrows']),
                                                                str(self.metadata['ncols']),
                                                                hex(id(self)))
        else:
            return "<Empty Raster object>"

    def close(self):
        """
        Close the GDAL raster file
        """
        self.datasource = None

    def read_array(self,
                   offsets=None,
                   band_order=None):
        """
        Method to read raster array with offsets and a specific band order
        Extracted array is 3 dimensional even if there is only one band specified: bands x rows x columns
        :param offsets: tuple or list - (xoffset, yoffset, xcount, ycount)
        :param band_order: order of bands to read
        """
        self.__init__(filename=self.filename,
                      datasource=self.datasource,
                      get_array=True,
                      offsets=offsets,
                      band_order=band_order)

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
            geom = ogr.CreateGeometryFromWkt(vector.wktlist[i])
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

    def get_bounds(self,
                   bounds=False,
                   bounds_coords=False,
                   bounds_wkt=False,
                   bounds_vector=True):
        """
        Method to extract boundary for a raster. This can be in the form of
        1) bounds (xmin, xmax, ymin, ymax)
        2) bound coords: list of closed ring polygon coords [(x1, y1), ...]
        3) wkt string of closed ring polygon
        4) Vector object

        :param bounds: boolean flag, if this is true subsequent flags are ignored (default: False)
        :param bounds_coords:boolean flag, if this is true subsequent flags are ignored (default: False)
        :param bounds_wkt: boolean flag, if this is true subsequent flags are ignored (default: False)
        :param bounds_vector: boolean flag (default: True)
        :return: bounds or bound_coords or wkt or Vector object
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

        wkt = "POLYGON(({}))".format(', '.join(list(' '.join([str(x), str(y)]) for (x, y) in bounds_list)))

        if bounds:
            return [mcx, mcx + ns * px, mcy - nl * py, mcy]

        elif bounds_coords:
            return bounds_list

        elif bounds_wkt:
            return wkt

        elif bounds_vector:

            vector = Vector()
            vector.wktlist.append(wkt)
            vector.spref_str = self.metadata['spref']
            vector.nfeat = 1
            vector.type = 3
            vector.name = str(os.path.basename(self.filename).split('.')[0]) + '_bounds'
            vector.spref = osr.SpatialReference()
            res = vector.spref.ImportFromWkt(vector.spref_str)

            memory_driver = ogr.GetDriverByName('Memory')
            vector.datasource = memory_driver.CreateDataSource('out')
            vector.layer = vector.datasource.CreateLayer('image_bounds',
                                                         srs=vector.spref,
                                                         geom_type=vector.type)

            fielddefn = ogr.FieldDefn('fid', ogr.OFTInteger)
            vector.fields.append(fielddefn)

            res = vector.layer.CreateField(fielddefn)

            layer_def = vector.layer.GetLayerDefn()
            geom = ogr.CreateGeometryFromWkt(wkt)
            feat = ogr.Feature(layer_def)
            feat.SetGeometry(geom)
            feat.SetField('fid', 0)

            vector.layer.CreateFeature(feat)
            vector.features = [feat]

            return vector

    def write_raster(self,
                     outfile,
                     file_type='GTiff'):
        """
        Method to write raster to disk
        :param outfile: Output file name
        :param file_type: File type (default: 'GTiff')
        :return: None
        """

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

            bandname = getattr(self.metadata, 'bandnames', None)

            if isinstance(self.metadata['bandnames'], list):
                bandname = str(self.metadata['bandnames'][i])
            elif type(bandname) == str:
                bandname = self.metadata['bandnames']

            if len(bandname) == 0 or bandname is None:
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
    OGR_FIELD_DEF = {
        'int': ogr.OFTInteger,
        'integer': ogr.OFTInteger,
        'long': ogr.OFTInteger,
        'float': ogr.OFTReal,
        'double': ogr.OFTReal,
        'str': ogr.OFTString,
        'string': ogr.OFTString,
        'bool': ogr.OFTInteger,
        'nonetype': ogr.OFSTNone,
        'none': ogr.OFSTNone
    }
    OGR_GEOM_DEF = {
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

    def __init__(self,
                 filename=None,
                 spref_str=None,
                 layer_index=0,
                 geom_type=3,
                 in_memory=False,
                 verbose=False,
                 feat_limit=None):
        """
        Constructor for class Vector
        :param filename: Name of the vector file (shapefile) with full path
        :param layer_index: Index of the vector layer to pull (default: 0)
        """

        self.filename = filename
        self.datasource = None

        self.features = list()
        self.attributes = list()
        self.wktlist = list()

        self.layer = None
        self.spref = None
        self.spref_str = None
        self.type = None

        self.name = 'Empty'
        self.nfeat = 0
        self.bounds = None
        self.fields = list()
        self.data = dict()

        if filename is not None and os.path.isfile(filename):

            # open vector file
            self.datasource = ogr.Open(self.filename)
            file_layer = self.datasource.GetLayerByIndex(layer_index)

            if in_memory:
                out_driver = ogr.GetDriverByName('Memory')
                out_datasource = out_driver.CreateDataSource('mem_source')
                self.layer = out_datasource.CopyLayer(file_layer, 'mem_source')
                self.datasource = out_datasource
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

            self.spref_str = self.spref.ExportToWkt()

            # other layer metadata
            self.type = self.layer.GetGeomType()
            self.name = self.layer.GetName()

            # number of features
            self.nfeat = self.layer.GetFeatureCount()

            if verbose:
                sys.stdout.write('Reading vector {} of type {} with {} features\n'.format(self.name,
                                                                                          str(self.type),
                                                                                          str(self.nfeat)))

            # get field defintions
            layer_definition = self.layer.GetLayerDefn()
            self.fields = [layer_definition.GetFieldDefn(i) for i in range(0, layer_definition.GetFieldCount())]

            # coordinates for bounds
            xmin = 0.0
            ymin = 0.0
            xmax = 0.0
            ymax = 0.0

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
                if feat_limit is not None:
                    if feat_count >= feat_limit:
                        break

                # extract feature attributes
                all_items = feat.items()

                # and feature geometry feature string
                geom = feat.GetGeometryRef()

                # close rings if polygon
                if geom_type == 3:
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

                # get bounds
                bounds_json = json.loads(geom.GetBoundary().ExportToJson())
                coord_list = bounds_json['coordinates']

                if bounds_json['type'] == 'LineString':
                    coords = coord_list

                elif bounds_json['type'] == 'MultiLineString':
                    coords = list()
                    for part in coord_list:
                        coords = coords + part
                else:
                    coords = [[0.0, 0.0]]

                xlist = list(coord[0] for coord in coords)
                ylist = list(coord[1] for coord in coords)

                # find bound coords
                if feat_count == 0:
                    xmin = min(xlist)
                    xmax = max(xlist)
                    ymin = min(ylist)
                    ymax = max(ylist)
                else:
                    if xmin > min(xlist):
                        xmin = min(xlist)
                    if xmax < max(xlist):
                        xmax = max(xlist)
                    if ymin > min(ylist):
                        ymin = min(ylist)
                    if ymax < max(ylist):
                        ymax = max(ylist)

                if verbose:
                    attr_dict = json.dumps(all_items)
                    sys.stdout.write('Feature {} of {} : attr {}\n'.format(str(feat_count+1),
                                                                           str(self.nfeat),
                                                                           attr_dict))

                self.attributes.append(all_items)
                self.features.append(new_feat)
                self.wktlist.append(geom.ExportToWkt())
                feat_count += 1

                feat = self.layer.GetNextFeature()

            self.nfeat = len(self.features)

            # get bounds geometry
            bound_coords = [[[xmin, ymin],
                             [xmin, ymax],
                             [xmax, ymax],
                             [xmax, ymin],
                             [xmin, ymin]]]

            self.bounds = ogr.CreateGeometryFromJson(json.dumps({"type": "Polygon",
                                                                 "coordinates": bound_coords}))

            if verbose:
                sys.stdout.write("\nInitialized Vector {} of type {} ".format(self.name,
                                                                              self.ogr_geom_type(self.type)) +
                                 "with {} feature(s) and {} attribute(s)\n\n".format(str(self.nfeat),
                                                                                   str(len(self.fields))))
        else:
            if in_memory:
                out_driver = ogr.GetDriverByName('Memory')
                out_datasource = out_driver.CreateDataSource('mem_source')
                self.datasource = out_datasource
                self.type = geom_type

                self.spref = osr.SpatialReference()
                res = self.spref.ImportFromWkt(spref_str)

                self.layer = self.datasource.CreateLayer('mem_layer',
                                                         srs=self.spref,
                                                         geom_type=geom_type)

                fid = ogr.FieldDefn('fid', ogr.OFTInteger)
                fid.SetPrecision()

                self.layer.CreateField(fid)
                self.fields = [fid]

            if verbose:
                sys.stdout.write("\nInitialized empty Vector\n")

    def __repr__(self):
        return "<Vector {} of type {} ".format(self.name,
                                               self.ogr_geom_type(self.type)) + \
               "with {} feature(s) and {} attribute(s)>".format(str(self.nfeat),
                                                                str(len(self.fields)))

    @staticmethod
    def ogr_data_type(x):
        """
        Method to get OGR data type, for use in creating OGR geometry fields
        :param x: Any data input
        :return: OGR data type
        """
        val = type(x).__name__.lower()

        try:
            return Vector.OGR_FIELD_DEF[val]
        except (KeyError, NameError):
            return Vector.OGR_FIELD_DEF['nonetype']

    @staticmethod
    def ogr_geom_type(x):
        """
        Method to return OGR geometry type from input string
        :param x: String to convert to OGR geometry type code
        :return: OGR geometry type code
        """

        if type(x).__name__ == 'str':
            comp_str = x.lower()

            try:
                return Vector.OGR_GEOM_DEF[comp_str]
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
                        val = str(x)
                    except:
                        val = None

            return Vector.ogr_data_type(val)

    def add_field(self,
                  field_name,
                  field_type,
                  **kwargs):

        """
        Function to add a field to a Vector object
        :param field_name: Name of the field (string)
        :param field_type: Type of the field
        :param kwargs: Keyword arguments: 1) precision: to be set when field type is float
                                          2) width: tp be set when field type is string
        :returns: None
        """

        if field_type in self.OGR_FIELD_DEF:
            field_type = self.OGR_FIELD_DEF[field_type]

        field = ogr.FieldDefn(field_name, field_type)

        if 'precision' in kwargs and field_type in (ogr.OFTReal, ogr.OFTInteger):
            field.SetPrecision(kwargs['precision'])

        if 'width' in kwargs and field_type == ogr.OFTString:
            field.SetWidth(kwargs['width'])

        self.layer.CreateField(field)
        self.fields += field

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
        self.wktlist.append(geom.ExportToWkt())
        if attr is not None:
            if primary_key is not None:
                attr.update({primary_key: self.nfeat})
            self.attributes.append(attr)
        elif primary_key is not None:
            self.attributes.append({primary_key: self.nfeat})

        self.nfeat += 1

    @staticmethod
    def get_osgeo_geom(geom_string,
                       geom_type='wkt'):
        """
        Method to return a osgeo geometry object
        :param geom_string: Wkt or json string
        :param geom_type: 'wkt', 'json', or 'wkb
        :return: osgeo geometry object
        """
        if geom_type == 'wkt':
            try:
                return ogr.CreateGeometryFromWkt(geom_string)
            except:
                return
        elif geom_type == 'json':
            try:
                return ogr.CreateGeometryFromJson(geom_string)
            except:
                return
        elif geom_type == 'wkb':
            try:
                return ogr.CreateGeometryFromWkb(geom_string)
            except:
                return
        else:
            raise ValueError("Unsupported geometry type")

    def merge(self,
              vector,
              remove=False):

        """
        Method to merge two alike vectors. This method only works for vectors
        that have same spref or spref_str, attribute keys, and geom types
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
            self.data.update(vector.data)

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

            out_vector.datasource = out_datasource
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
                                                   srs=self.spref,
                                                   geom_type=self.type)

            for field in self.fields:
                out_layer.CreateField(field)

            layer_defn = out_layer.GetLayerDefn()

            if len(self.wktlist) > 0:
                for i, wkt_geom in enumerate(self.wktlist):
                    geom = ogr.CreateGeometryFromWkt(wkt_geom)
                    feat = ogr.Feature(layer_defn)
                    feat.SetGeometry(geom)

                    for attr, val in self.attributes[i].items():
                        feat.SetField(attr, val)

                    out_layer.CreateFeature(feat)

            elif len(self.features) > 0:
                for feature in self.features:
                    out_layer.CreateFeature(feature)

            else:
                sys.stdout.write('No features found... closing file.\n')

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
            out_vector.datasource = temp_datasource
            out_vector.wktlist = list()

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
                out_vector.wktlist.append(temp_geom.ExportToWkt())

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
        vector.spref_str = destination_spatial_ref.ExportToWkt()

        # get source spatial reference from Spatial reference WKT string in self
        source_spatial_ref = self.spref

        # create a transform tool (or driver)
        transform_tool = osr.CoordinateTransformation(source_spatial_ref,
                                                      destination_spatial_ref)

        # Create a memory layer
        memory_driver = ogr.GetDriverByName('Memory')
        vector.datasource = memory_driver.CreateDataSource('out')

        # create a layer in memory
        vector.layer = vector.datasource.CreateLayer('temp',
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

        vector.wktlist = list()
        vector.attributes = self.attributes

        # convert each feature
        for feat in self.features:

            # transform geometry
            temp_geom = feat.GetGeometryRef()
            temp_geom.Transform(transform_tool)

            vector.wktlist.append(temp_geom.ExportToWkt())

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
            self.datasource = vector.datasource
            self.wktlist = vector.wktlist
            self.spref_str = vector.spref_str

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

