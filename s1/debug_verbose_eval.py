"""
Debug runner: load a MAPPO checkpoint and run one episode with verbose reward logging.

Produces a CSV `debug_rewards.csv` with columns:
  step, agent_idx, junction_id, reward, episode_reward

Usage:
  python debug_verbose_eval.py --checkpoint <path/to/checkpoint> --scenario event --steps 21600

This script is non-intrusive and only reads the environment and models.
"""
import argparse
import csv
import os
import time
from mappo_k1_implementation import MAPPOConfig, MAPPOAgent, K1Environment


def run_debug(checkpoint, scenario, steps, out_csv, threshold=50.0):
    config = MAPPOConfig()
    # Map scenario to config (recognized names -> .sumocfg files)
    scenario_map = {
        'event': 'k1_6h_event.sumocfg',
        'weekday': 'k1_6h_weekday.sumocfg',
        'weekend': 'k1_6h_weekend.sumocfg',
        'event_repro': 'k1_6h_event_repro.sumocfg',
        'gridlock': 'k1_6h_gridlock.sumocfg',
        'incident': 'k1_6h_incident.sumocfg',
        'spike': 'k1_6h_spike.sumocfg',
        'night_surge': 'k1_6h_night_surge.sumocfg',
    }
    if scenario in scenario_map:
        config.SUMO_CONFIG = scenario_map[scenario]
    else:
        # Assume it's a direct filename (just the basename, not a full path)
        config.SUMO_CONFIG = os.path.basename(scenario)
    config.STEPS_PER_EPISODE = steps

    agent = MAPPOAgent(config)
    # Load checkpoint models (weights only)
    if checkpoint:
        print(f"Loading models from: {checkpoint}")
        try:
            agent.load_models(checkpoint)
        except Exception as e:
            print(f"Failed to load models from {checkpoint}: {e}")

    env = K1Environment(config)

    # Reset env
    local_states, global_state = env.reset()

    episode_reward = 0.0

    # Prepare CSV
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    csv_file = open(out_csv, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'agent_idx', 'junction_id', 'reward', 'episode_reward'])

    try:
        for step in range(steps):
            actions, log_probs, entropies = agent.select_actions(local_states)
            next_local_states, next_global_state, rewards, done = env.step(actions)

            # Log per-agent rewards
            for i, r in enumerate(rewards):
                writer.writerow([step, i, config.JUNCTION_IDS[i], float(r), episode_reward + float(sum(rewards))])

            episode_reward += sum(rewards)

            # Print summary every 500 steps and flag large values
            if step % 500 == 0:
                print(f"Step {step}/{steps} | cumulative reward: {episode_reward:.2f}")
            for i, r in enumerate(rewards):
                if abs(r) >= threshold:
                    print(f"⚠ Large reward at step {step} agent {i} ({config.JUNCTION_IDS[i]}): {r}")

            local_states = next_local_states
            global_state = next_global_state

            if done:
                print(f"Environment signaled done at step {step}")
                break

    finally:
        csv_file.close()
        env.close()

    print(f"Debug run finished. CSV written to: {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=False, default=None)
    parser.add_argument('--scenario', type=str, default='event')
    parser.add_argument('--steps', type=int, default=21600)
    parser.add_argument('--out', type=str, default='s1/debug_rewards.csv')
    parser.add_argument('--threshold', type=float, default=50.0,
                        help='Threshold to print large reward warnings (per-agent)')
    args = parser.parse_args()

    run_debug(args.checkpoint, args.scenario, args.steps, args.out, threshold=args.threshold)
