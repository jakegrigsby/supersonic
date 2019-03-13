import os
os.chdir('..')
import argparse

from supersonic import agent, camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl', type=str)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    sonic = agent.BaseAgent(args.lvl) 
    if args.weights:
        sonic.load_weights(args.weights)
    cam = camera.Camera(sonic)
    record_path = 'logs/prototype_agent_00/gameplay/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    cam.start_recording(os.path.join(record_path, 'testvid'))
    sonic.test(1, render=True, max_ep_steps=100)
    cam.stop_recording()