import sys
import argparse
import datetime
import re

import matplotlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",help = "input file")
    parser.add_argument("-o",help = "output file")
    args = parser.parse_args()

    i_file = "log.txt"
    o_file = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    if args.i != None:
        i_file = args.i
    
    if args.o != None:
        o_file = args.o
    
    with open(i_file) as fp_log:
        r_reward = []
        d_reward = []
        
        reward_matcher = re.compile("total reward (*) (*)")
        #reward_matcher


