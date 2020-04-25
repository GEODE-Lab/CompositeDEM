import pickle
import datetime
import sys
import os
import psutil
from demLib.common import Timer





if __name__ == '__main__':

    script, pickle_file, outdir = sys.argv

    t1 = datetime.datetime.now()

    with open(pickle_file, 'rb') as fileptr:
        funky_list = pickle.load(fileptr)

    print(len(funky_list))

    print_memory_usage()

    outfile = outdir + 'funky_list.txt'

    with open(outfile, 'w') as fileptr:
        for elem in funky_list:
            fileptr.write('[' + ','.join(list(str(elem_part) for elem_part in elem)) + ']\n')

    t2 = datetime.datetime.now()

    print_memory_usage()

    print('Time taken: {}'.format(Timer.display_time(t2-t1)))


