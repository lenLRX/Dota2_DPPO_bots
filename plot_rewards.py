import sys
import argparse
import datetime
import re
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",help = "input file")
    parser.add_argument("-o",help = "output file")
    args = parser.parse_args()

    i_file = "log.txt"
    o_file = "./log/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S") + "-reward.png"

    if args.i != None:
        i_file = args.i
    
    if args.o != None:
        o_file = args.o
    
    with open(i_file) as fp_log:
        reward = []
        
        reward_matcher = re.compile(".*loss (.*)")
        for line in fp_log:
            ret = reward_matcher.match(line)
            if ret is not None:
                ret = ret.groups()
                reward.append(float(ret[0]))

    plt.figure()
    plt.plot(list(range(len(reward))),reward)
    plt.savefig(o_file)
    print(os.path.abspath(o_file))

