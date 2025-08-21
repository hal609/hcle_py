"""A GymNESium environment for generic."""

import numpy as np
from ..nes_env import *

# Possible inputs comprise all individual button presses as well a
# any direction + A or B and A+B simultaneously
inputs = [0,1,2,4,5,6,8,9,10,16,32,64,65,66,68,72,128,129,130,132,136,192]
    
class WeightedObjectiveReward:
    def __init__(self, objectives_file):
        self.objectives = self.load_objectives(objectives_file)

    def load_objectives(self, filename):
        objectives = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                weight = float(parts[0])
                indices = list(map(int, parts[1:]))
                objectives.append((weight, indices))
        return objectives
    
    
    def extract_objective_value(self, ram_snapshot, objective_indices):
        """Extracts a tuple of values from a RAM snapshot for a given objective."""
        return tuple(ram_snapshot[i] for i in objective_indices)

    def get_reward(self, current_ram, previous_ram):
        """
        Calculates the reward for a single transition based on lexicographic change.

        Args:
            current_ram: The RAM snapshot after the action was taken.
            previous_ram: The RAM snapshot before the action was taken.
            objectives: The list of (weight, indices) tuples.

        Returns:
            A float representing the total reward for this step.
        """
        total_reward = 0.0

        if previous_ram is None:
            return 0.0 # No reward on the very first frame

        for weight, objective_indices in self.objectives:
            current_value = self.extract_objective_value(current_ram, objective_indices)
            previous_value = self.extract_objective_value(previous_ram, objective_indices)

            if current_value > previous_value:
                # Progress was made
                total_reward += weight
            elif current_value < previous_value:
                # Regressed
                total_reward -= weight
            # If equal, do nothing (reward is 0 for this objective)

        return total_reward
    
class GenericEnv(NESEnv):
    """An environment for playing an NES game with OpenAI Gym."""

    def __init__(self, game_name:str, objective_file:str = None, render_mode:str = "rgb_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Initialize a new environment.

        Args:
            rom_path (str): The path to the ROM file.
            render_mode (str): Optional - Either "rgb_array" or "human"  which defines whether the environment should display a window for each env.
            fps_limit (int): The frame rate limit of the game, negative values are unlimited. Defaults to -1
        
        Returns:
            None
        """
        super().__init__(game_name, render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)
        self.setActions(inputs)

        if objective_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            objective_file = os.path.join(current_dir, '..', 'objectives', f'{game_name}.objectives')
        
        if not os.path.exists(objective_file):
            raise FileExistsError(f"No objective file found at: '{objective_file}'.")
        
        self.objReward = WeightedObjectiveReward(objective_file)

        self.reward_window = []
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = self.objReward.get_reward(self.current_ram, self.previous_ram)
        # self.reward_window.append(reward)
        # if len(self.reward_window) > 60: self.reward_window.pop(0)
        # print(f"{int(np.array(self.reward_window).mean()):02d}")
        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return False