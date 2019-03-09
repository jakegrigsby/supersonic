import os
os.chdir('..')
import argparse

from supersonic import agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl', type=str)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    sonic = agent.BaseAgent(args.lvl) 
    if args.weights:
        sonic.load_weights(args.weights)
    sonic.test(1, render=True)