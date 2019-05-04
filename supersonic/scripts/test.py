import os
os.chdir('..')
import argparse

from supersonic import ppo, camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lvl', default="GreenHillZone.Act1", type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--record', default=False, type=bool)
    parser.add_argument('--record_path', default='logs/gameplay', type=str)
    args = parser.parse_args()

    sonic = ppo.PPOAgent(args.lvl) 
    if args.weights:
        sonic.load_weights(args.weights)
    if args.record:
        cam = camera.Camera(sonic)
        record_path = args.record_path
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        cam.start_recording(os.path.join(record_path, 'testvid'))
    sonic.test(1, render=True, max_ep_steps=4500)
    if args.record:
        cam.stop_recording()
