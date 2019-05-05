import os
os.chdir('..')
import argparse
import supersonic.ppo as ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl', default="GreenHillZone.Act1", type=str)
    parser.add_argument('--logdir', default="agent", type=str)
    parser.add_argument('--rollouts', default=30000, type=int)
    parser.add_argument('--render', default=0, type=int)
    args = parser.parse_args()
    sonic = ppo.PPOAgent(args.lvl, log_dir=args.logdir)
    sonic.train(args.rollouts, render=args.render-1)
