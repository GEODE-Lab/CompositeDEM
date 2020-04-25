import os
import sys
import time
import shutil
import psutil
import fnmatch
import operator
import itertools
from functools import wraps
from itertools import takewhile, repeat


__all__ = ['Common',
           'File',
           'Timer']


class Common(object):

    @staticmethod
    def string_to_type(x):
        """
        Method to return name of the data type
        :param x: input item
        :return: string
        """
        if type(x).__name__ == 'str':
            x = x.strip()
            try:
                val = int(x)
            except (ValueError, TypeError):
                try:
                    val = float(x)
                except (ValueError, TypeError):
                    try:
                        val = str(x)
                    except (ValueError, TypeError):
                        val = None
            x = val
        return x

    @staticmethod
    def group_multi(in_list):
        """
        Method to group all numbers that occur together in any piece-wise-list manner or
        in other words find all connected components in a graph

        example: input [[2,4],[5,6,7,8,10],[3,9,12,4],[14,12],[99,100,101],[104,3],[405,455,456],[302,986,2]]
        will be grouped to [[2, 3, 4, 9, 12, 14, 104, 302, 986],[5, 6, 7, 8, 10],[99, 100, 101],[405, 455, 456]]

        :param in_list: List of lists
        :return: list of lists
        """
        out_list = []
        while len(in_list) > 0:
            first_chunk, *rest_chunks = in_list
            first_chunk = set(first_chunk)

            size_first = -1
            while len(first_chunk) > size_first:
                size_first = len(first_chunk)

                other = []
                for chunk in rest_chunks:
                    if len(first_chunk.intersection(set(chunk))) > 0:
                        first_chunk |= set(chunk)
                    else:
                        other.append(chunk)
                rest_chunks = other

            out_list.append(list(first_chunk))
            in_list = rest_chunks

        return out_list

    @staticmethod
    def group_consecutive(arr):
        """
        Method to group consecutive elements into one sorted list
        :param arr: List of numbers
        :returns: List of lists
        """
        grouped_elements = []
        for _, group in itertools.groupby(enumerate(sorted(arr)), key=lambda x: x[0] - x[1]):
            grouped_elements.append(sorted(list(map(operator.itemgetter(1), group))))
        return grouped_elements

    @staticmethod
    def get_memory_usage():
        """
        Function to return memory usage of the python process
        :returns: String of the form '16.2 GB'
        """
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss

        if 2 ** 10 <= mem < 2 ** 20:
            div = float(2 ** 10)
            suff = ' KB'
        elif 2 ** 20 <= mem < 2 ** 30:
            div = float(2 ** 20)
            suff = ' MB'
        elif 2 ** 30 <= mem < 2 ** 40:
            div = float(2 ** 30)
            suff = ' GB'
        elif mem >= 2 ** 40:
            div = float(2 ** 40)
            suff = ' TB'
        else:
            div = 1.0
            suff = ' BYTES'

        return '{:{w}.{p}f}'.format(process.memory_info().rss / div, w=5, p=1) + suff

    @staticmethod
    def cprint(text,
               newline='\n'):
        """
        Print to stdout and flush
        :param text: Text to print to stdout
        :param newline: Newline character, default '\n' but can be changed
                        to '' to concatenate consequetive outputs
        """
        sys.stdout.write(str(text) + newline)
        sys.stdout.flush()


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
            # sys.stdout.write('File exists: ' + filename)
            # sys.stdout.write('Deleting file...')
            try:
                os.remove(self.filename)
            except OSError:
                sys.stdout.write('File already exists. Error deleting file!')
                components = self.basename.split('.')
                if len(components) < 2:
                    self.filename = self.dirpath + self.sep + self.basename + '_' + str(counter)
                else:
                    self.filename = self.dirpath + self.sep + ''.join(components[0:-1]) + \
                               '_(' + str(counter) + ').' + components[-1]
                # sys.stdout.write('Unable to delete, using: ' + filename)
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

    def file_lines(self,
                   nlines=False,
                   bufsize=102400):
        """
        Find number of lines or get text lines in a text or csv file
        :param nlines: If only the number of lines in a file should be returned
        :param bufsize: size of buffer to be read
        :return: list or number
        """
        with open(self.filename, 'r') as f:
            bufgen = takewhile(lambda x: x, (f.read(bufsize) for _ in repeat(None)))

            if nlines:
                val = sum(buf.count('\n') for buf in bufgen if buf)
            else:
                val = list()
                remaining = ''
                for buf in bufgen:
                    if buf:
                        temp_lines = (remaining + buf).split('\n')
                        if len(temp_lines) <= 1:
                            remaining += ''.join(temp_lines)
                        else:
                            val += temp_lines[:-1]
                            remaining = temp_lines[-1]
        return val


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
                sys.stdout.write("Type not coercible to Float")

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

                sys.stdout.write("Time it took to run {}: {}\n".format(func.__name__,
                                                                       t))
                return val

            return wrapper

        return time_it

