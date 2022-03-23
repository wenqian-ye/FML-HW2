#!/usr/bin/env python

import os, sys, math, random
from collections import defaultdict

if sys.version_info[0] >= 3:
    xrange = range

def process_options(argv):
    argc = len(argv)   
    dataset = argv[1]
    new_file = argv[2]

    return dataset, new_file

def main(argv=sys.argv):
    dataset, new_file = process_options(argv)

    l = sum(1 for line in open(dataset,'r'))
    dataset = open(dataset,'r')
    new_file = open(new_file,'w')
    for i in xrange(l):
        line = dataset.readline()
        line_split = line.split(' ')
        label = int(line_split[0])
        if label <= 9:
            line_split[0] = "0"
            new_line = " ".join(line_split)
            new_file.write(new_line)
        else:
            line_split[0] = "1"
            new_line = " ".join(line_split)
            new_file.write(new_line) 
    new_file.close()
    dataset.close()

if __name__ == '__main__':
    main(sys.argv)