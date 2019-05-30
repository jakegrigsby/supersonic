import os

os.chdir("..")
import argparse

from supersonic import ppo, camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvl", default="GreenHillZone.Act1", type=str)
    parser.add_argument("--weights", default="GreenHillZoneAct1/checkpoint_9500", type=str)
    parser.add_argument("--record", default=False, type=bool)
    parser.add_argument("--filename", default="test_clip", type=str)
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--render", default=True)
    args = parser.parse_args()

    sonic = ppo.PPOAgent(args.lvl)
    if args.weights:
        weights_path = os.path.join("weights/", args.weights)
        sonic.load_weights(weights_path)
    if args.record:
        cam = camera.Camera(sonic, highlight_buffer_capacity=0)
        record_path = f"data/gameplay_footage/"
        filename = f"{args.filename}.mp4"
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        cam.start_recording(os.path.join(record_path, filename))
    sonic.test(args.episodes, render=args.render, max_ep_steps=2500)
    if args.record:
        cam.stop_recording()
