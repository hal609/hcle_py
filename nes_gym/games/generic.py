"""A GymNESium environment for generic."""

import numpy as np
from ..nes_env import *

# Possible inputs comprise all individual button presses as well a
# any direction + A or B and A+B simultaneously
inputs = [0,1,2,4,5,6,8,9,10,16,32,64,65,66,68,72,128,129,130,132,136,192]
    
class GenericEnv(NESEnv):
    """An environment for playing an NES game with OpenAI Gym."""

    def __init__(self, render_mode:str = "rgb_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Initialize a new environment.

        Args:
            rom_path (str): The path to the ROM file.
            render_mode (str): Optional - Either "rgb_array" or "human"  which defines whether the environment should display a window for each env.
            fps_limit (int): The frame rate limit of the game, negative values are unlimited. Defaults to -1
        
        Returns:
            None
        """
        super().__init__("generic", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)
        self.setActions(inputs)

    def _did_step(self) -> None:
        self.episode_frame_count += 1
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = 0.01
        reward += float(self.get_score_change())

        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return False