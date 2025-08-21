# hcle_py/envs/smb1.py

"""GymNESium environment for SMB1"""

import numpy as np
from ..nes_env import *

inputs = [
    # NES_INPUT_NONE,
    # NES_INPUT_LEFT,
    # NES_INPUT_RIGHT,
    # NES_INPUT_A,
    # NES_INPUT_RIGHT | NES_INPUT_A, 
    NES_INPUT_RIGHT | NES_INPUT_B,
    NES_INPUT_RIGHT | NES_INPUT_B | NES_INPUT_A, 
    ]

PLAYER_STATE = 0x000E
GAME_MODE = 0x0770 # 00 - Start demo, 01 - Start normal, 02 - End current world, 03 - End game (dead)
ON_PRELEVEL = 0x075E
LEVEL_LOADING = 0x0772
Y_VIEWPORT = 0x00B5 # Greater than 1 means off screen
CURRENT_PAGE = 0x006D
X_POS = 0x0086
    
class SMB1Env(NESEnv):
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
        super().__init__("smb1", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

    @property
    def _in_game(self) :
        '''Return the current game mode.'''
        return self.current_ram[LEVEL_LOADING] == 3 and self.current_ram[GAME_MODE] != 0
    
    @property
    def _is_dead(self):
        return self.current_ram[PLAYER_STATE] == 0x0B or self.current_ram[Y_VIEWPORT] > 0x1
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        while (not self._in_game):
            self._frame_advance(0)
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(NES_INPUT_START)

    def _will_step(self):
        if not self._in_game: self.skip_between_rounds()

    def _did_step(self):
        # If match has started and no save exists, make one
        if self._has_backup: return
        if not self._in_game: return
        self._backup()

    def get_mario_pos(self):
        first_digit = np.uint64(self.current_ram[CURRENT_PAGE]) * 0x100

        return first_digit + np.uint64(self.current_ram[X_POS])
    
    def get_mario_pre_pos(self):
        first_digit = np.uint64(self.previous_ram[CURRENT_PAGE]) * 0x100

        return first_digit + np.uint64(self.previous_ram[X_POS])
    
    def get_pos_change(self) -> int:
        page_dif = int(self.current_ram[CURRENT_PAGE]) - int(self.previous_ram[CURRENT_PAGE])
        x_dif = int(self.current_ram[X_POS]) - int(self.previous_ram[X_POS])

        return int(page_dif) * 0x100 + x_dif
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = -0.1
        self.value_change(CURRENT_PAGE)
        reward += float(self.get_pos_change())

        is_dead = self._is_dead
        if is_dead: reward = -20
        return reward
        

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self._is_dead