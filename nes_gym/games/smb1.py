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

        self.ram_dict = {
            "current_page": 0x006D,
            "x_pos": 0x0086,
            "level": 0x0760,
            "world": 0x075F,
            "player_state": 0x000E
        }

        self.player_states = {
            "Player dies": 0x06,
            "Entering area": 0x07,
            "Normal": 0x08
        }

        self.reward_hist = []

    @property
    def _in_game(self) :
        '''Return the current game mode.'''
        return self.nes[0x0770] == 0x01
    
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
        self.episode_frame_count += 1

        if self._has_backup: return
        if self._in_game: return
        self._backup()

    def get_mario_pos(self):
        first_digit = np.uint64(self.current_ram[self.ram_dict["current_page"]]) * 0x100

        return first_digit + np.uint64(self.current_ram[self.ram_dict["x_pos"]])
    
    def get_mario_pre_pos(self):
        first_digit = np.uint64(self.previous_ram[self.ram_dict["current_page"]]) * 0x100

        return first_digit + np.uint64(self.previous_ram[self.ram_dict["x_pos"]])
    
    def get_pos_change(self) -> int:
        page_dif = int(self.current_ram[self.ram_dict["current_page"]]) - int(self.previous_ram[self.ram_dict["current_page"]])
        x_dif = int(self.current_ram[self.ram_dict["x_pos"]]) - int(self.previous_ram[self.ram_dict["x_pos"]])

        return int(page_dif) * 0x100 + x_dif
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = -0.1
        self.value_change(self.ram_dict["current_page"])
        reward += float(self.get_pos_change())

        is_dead = (self.current_ram[self.ram_dict["player_state"]] == self.player_states["Player dies"])
        if is_dead: reward = -20
        return reward
        

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        is_dead = (self.current_ram[self.ram_dict["player_state"]] == self.player_states["Player dies"])
        return is_dead