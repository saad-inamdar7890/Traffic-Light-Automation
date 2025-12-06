"""
train_event_repro.py
--------------------
Simple wrapper to train MAPPO on the generated `k1_6h_event_repro.sumocfg` scenario.

Usage:
  python train_event_repro.py --resume-checkpoint <path> --max-hours 1 --num-episodes 10

This imports the training function and runs directly with a modified config.
"""
import argparse
import os
from mappo_k1_implementation import MAPPOConfig, train_mappo


def main(resume_checkpoint=None, max_hours=None, num_episodes=None):
    config = MAPPOConfig()
    config.SUMO_CONFIG = 'k1_6h_event_repro.sumocfg'
    config.STEPS_PER_EPISODE = 21600
    if num_episodes is not None:
        config.NUM_EPISODES = num_episodes

    # Ensure model/log dirs exist
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    train_mappo(config, resume_checkpoint=resume_checkpoint, max_hours=max_hours)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    parser.add_argument('--max-hours', type=float, default=None)
    parser.add_argument('--num-episodes', type=int, default=None)
    args = parser.parse_args()

    main(resume_checkpoint=args.resume_checkpoint, max_hours=args.max_hours, num_episodes=args.num_episodes)
