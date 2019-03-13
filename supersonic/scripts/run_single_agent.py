import os
os.chdir('..')
import supersonic.agent as agent

if __name__ == "__main__":
    sonic = agent.BaseAgent('GreenHillZone.Act1', log_dir='prototype_agent')
    sonic.train(100000, device='/gpu:0', render=False)