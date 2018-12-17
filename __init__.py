from common import Raster, Vector, _find_dict_index


class FieldError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 401

    def __init__(self,
                 error="Bad Request",
                 description="Missing or invalid information",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class SizeError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 402

    def __init__(self,
                 error=None,
                 description="Improper substitution or value",
                 status_code=400,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class UninitializedError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 403

    def __init__(self, error=None,
                 description="Object not initialized",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class ObjectNotFound(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 404

    def __init__(self, error=None,
                 description="Object not found",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class ImageProcessingError(Exception):
    status_code = 501

    def __init__(self, error="Image Processing Error",
                 description="Something went wrong processing your image",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class TileNotFound(Exception):
    status_code = 601

    def __init__(self, error="Image Processing Error",
                 description="Image tile not found",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class FileNotFound(Exception):
    status_code = 701

    def __init__(self, error=None,
                 description="No matching file found on disk",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error
