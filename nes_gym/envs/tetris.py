"""GymNESium environment for SMB1"""

import numpy as np
from ..nes_env import *

inputs = [
    NES_INPUT_NONE,
    NES_INPUT_LEFT,
    NES_INPUT_DOWN,
    NES_INPUT_RIGHT,
    NES_INPUT_A
    ]
    
class TetrisEnv(NESEnv):
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
        super().__init__("tetris", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

        self.ram_dict = {
            "game_phase": 0x0048,
            "score_bytes": [0x0053, 0x0054, 0x0055],
            "score_third_byte": 0x0053,
            "score_second_byte": 0x0054,
            "score_first_byte": 0x0055,
            "rng1": 0x0017,
            "rng2": 0x0018,
            "game_over": 0x0058, # = 0xA at finish
        }

    @property
    def _in_game(self) :
        '''Return the current game mode.'''
        # return True
        return self.current_ram[self.ram_dict["game_phase"]] != 0
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        while (not self._in_game):
            self._frame_advance(0)
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(NES_INPUT_START)

    def get_current_score(self) -> np.int64:
        return np.int64(f"{self.read_mult_byte(self.ram_dict["score_bytes"], endian="little"):x}")
    
    def get_previous_score(self) -> np.int64:
        return np.int64(f"{self.read_mult_byte(self.ram_dict["score_bytes"], endian="little", ram_selection=self.previous_ram):x}")
    
    def get_score_change(self) -> np.int64:
        return self.get_current_score() - self.get_previous_score()
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = 0.01
        score_change = max(float(self.get_score_change()), 0)
        reward += score_change
        if self.current_ram[self.ram_dict["game_over"]] == 0xA: reward -= 20
        reward = reward / 1e4
        
        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self.current_ram[self.ram_dict["game_over"]] == 0xA