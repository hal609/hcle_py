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
# inputs = np.arange(255)
    
class DrMarioEnv(NESEnv):
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
        super().__init__("drmario", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

        self.ram_dict = {
            "game_phase": 0x0048,
            "score_bytes": [0x072E, 0x072D, 0x072C, 0x072B, 0x072A, 0x0729, 0x0728],
            "level_num": 0x0316,
            "mode": 0x0046,
            "virus_level": 0x0096,
            "pill_speed": 0x030B,
            "frames_til_drop": 0x0312,
            "game_over": 0x0058, # = 0xA at finish
        }

        self.til_drop_count = 0

    @property
    def _in_game(self) :
        '''Return the current game mode.'''
        return self.current_ram[self.ram_dict["mode"]] > 1
    
    @property
    def _playing(self) :
        '''Return the current game mode.'''
        return self.current_ram[self.ram_dict["mode"]] == 4
    
    @property
    def _game_over(self) :
        '''Return the current game mode.'''
        return self.current_ram[self.ram_dict["mode"]] == 7
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        # return
        while (not self._in_game):
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(0)
        while (not self._playing):
            self._frame_advance(0)

    def get_current_score(self) -> np.int64:
        digits = [f"{self.current_ram[loc]}" for loc in self.ram_dict["score_bytes"]]
        return np.int64("".join(digits))
    
    def get_previous_score(self) -> np.int64:
        digits = [f"{self.previous_ram[loc]}" for loc in self.ram_dict["score_bytes"]]
        return np.int64("".join(digits))
    
    def get_score_change(self) -> np.int64:
        return self.get_current_score() - self.get_previous_score()
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = 0.01
        reward += float(self.get_score_change()) 
        if self._game_over: reward -= 20
        reward /= 1e3

        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self._game_over