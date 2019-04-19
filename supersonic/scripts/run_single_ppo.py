import os
os.chdir('..')
import argparse
import supersonic.ppo as ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl', default="GreenHillZone.Act1", type=str)
    parser.add_argument('--logdir', default="agent", type=str)
    args = parser.parse_args()
    sonic = ppo.PPOAgent(args.lvl, log_dir=args.logdir)
    sonic.train(30000, device='/cpu:0', render=True)
