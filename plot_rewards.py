import sys
import argparse
import datetime
import re

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
        r_reward = []
        d_reward = []
        
        reward_matcher = re.compile("total reward (.*) (.*)")
        for line in fp_log:
            ret = reward_matcher.match(line)
            if ret is not None:
                ret = ret.groups()
                r_reward.append(float(ret[0]))
                d_reward.append(float(ret[1]))

    plt.figure()
    plt.plot(list(range(len(r_reward))),r_reward)
    plt.plot(list(range(len(r_reward))),d_reward)
    plt.savefig(o_file)

