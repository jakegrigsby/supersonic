import os
os.chdir('..')
import supersonic.ppo as ppo

if __name__ == "__main__":
    sonic = ppo.PPOAgent('GreenHillZone.Act1', log_dir='sonic3')
    sonic.train(30000, device='/cpu:0', render=True)
