import os
os.chdir('..')
import supersonic.agent as agent

if __name__ == "__main__":
    sonic = agent.BaseAgent('GreenHillZone.Act1', log_dir='firstTrylogs')
    sonic.train(1000, device='/cpu:0')
    sonic.save_weights('model_zoo/firstTryweights')

