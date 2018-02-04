#coding=utf-8
import sys
import argparse
import datetime
import re
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",help = "input file")
    parser.add_argument("-o",help = "output file")
    args = parser.parse_args()

    i_file = "log.txt"
    o_file = "./log/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S") + "-ratio.png"

    if args.i != None:
        i_file = args.i
    
    if args.o != None:
        o_file = args.o
    
    with open(i_file) as fp_log:
        ratios = []
        
        reward_matcher = re.compile(".*ratio (.*)")
        for line in fp_log:
            ret = reward_matcher.match(line)
            if ret is not None:
                ret = ret.groups()
                ratios.append(eval(ret[0]))

    plt.figure()
    lines = plt.plot(np.asarray(ratios))
    plt.legend(lines,["None","move","attack"])
    plt.savefig(o_file)
    print(os.path.abspath(o_file))

