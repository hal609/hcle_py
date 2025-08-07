import gymnasium as gym
import nes_gym
import numpy as np
import math

def create_env(game_name: str, render_mode: str = "rgb_array"):
    """Creates a single environment instance."""
    env = gym.make(f'NES/{game_name}-v1', render_mode=render_mode)
    return env

if __name__ == '__main__':
    game = "SuperMarioBros"
    lookahead_frames = 6
    frame_skip = 8
    gamma = 0.99
    num_worker_envs = 32

    print("Creating main environment...")
    main_env = create_env(game, render_mode="human")
    action_space_size = main_env.action_space.n

    print(f"Creating {num_worker_envs} parallel search environments...")
    search_vec_env = gym.vector.SyncVectorEnv(
        [lambda: create_env(game) for _ in range(num_worker_envs)]
    )

    print(f"Each step will be performed in {math.ceil((action_space_size**lookahead_frames) / num_worker_envs)} batch(es).")
    print(f"Final batch has {(action_space_size**lookahead_frames) % num_worker_envs} redundant environments.")

    search_vec_env.call("reset", savestate=None)
    main_env.reset()

    action_list = []

    while True:
        current_state = main_env.unwrapped.create_savestate()

        # Initialize the search tree
        #   Each item tracks: [savestate, cumulative_reward, first_action]
        paths = [[current_state, 0.0, None]]

        # Search
        for k in range(lookahead_frames):
            # Expand the tree: determine all paths for the next level
            next_level_tasks = []
            for path_state, path_reward, first_action in paths:
                for action in range(action_space_size):
                    # For the first level, the first_action is the action itself
                    current_first_action = first_action if first_action is not None else action
                    next_level_tasks.append([path_state, path_reward, current_first_action, action])
            
            # Process all tasks for this level in batches
            all_results = []
            num_batches = math.ceil(len(next_level_tasks) / num_worker_envs)

            for i in range(num_batches):
                batch_start = i * num_worker_envs
                batch_end = min(batch_start + num_worker_envs, len(next_level_tasks))
                batch = next_level_tasks[batch_start:batch_end]
                
                # Prepare the batch for the vector environment
                batch_size = len(batch)
                batch_states = [task[0] for task in batch]
                batch_actions = [task[3] for task in batch]

                actions_to_step = np.zeros(num_worker_envs, dtype=int)
                actions_to_step[:batch_size] = batch_actions
                
                # Load the unique states for this batch into the workers
                for j in range(batch_size):
                    search_vec_env.envs[j].reset(savestate=batch_states[j])

                # Execute the frame-skipped step for the batch
                step_rewards = np.zeros(batch_size)
                active_mask = np.ones(batch_size, dtype=bool)
                for _ in range(frame_skip):
                    # We only care about the first `batch_size` workers for this batch
                    _obs, rewards, dones, truns, _infos = search_vec_env.step(actions_to_step)
                    step_rewards += rewards[:batch_size] * active_mask
                    active_mask &= ~(dones[:batch_size] | truns[:batch_size])
                
                # Get the resulting savestates
                new_states = [search_vec_env.envs[j].unwrapped.create_savestate() for j in range(batch_size)]
                
                # Store the results for this batch
                for j in range(batch_size):
                    initial_reward = batch[j][1]
                    first_action = batch[j][2]
                    new_total_reward = initial_reward + (gamma ** k) * step_rewards[j]
                    all_results.append([new_states[j], new_total_reward, first_action])
            
            # The results of this level become the paths for the next level
            paths = all_results

        # Find the best first action from all the completed paths
        #    The `paths` list contains all N^K final results.
        final_rewards = np.array([p[1] for p in paths])
        best_path_indices = np.where(final_rewards == final_rewards.max())[0]
        chosen_path_index = np.random.choice(best_path_indices)
        
        chosen_action = paths[chosen_path_index][2]

        print(f"Best {lookahead_frames}-step reward: {final_rewards.max():.2f}. Chosen Action: {chosen_action}")

        # _obs, _reward, terminated, truncated, _info = main_env.step(chosen_action)
        # action_list.append(chosen_action)
        # np.save(f"{game}_actions_look{lookahead_frames}_skip{frame_skip}.npy", np.array(action_list, dtype=np.uint8))

        # Execute chosen action in main environment
        for _ in range(frame_skip):
            _obs, _reward, terminated, truncated, _info = main_env.step(chosen_action)
            action_list.append(chosen_action)
            np.save(f"{game}_actions_look{lookahead_frames}_skip{frame_skip}.npy", np.array(action_list, dtype=np.uint8))
            if terminated or truncated:
                main_env.reset()
                break