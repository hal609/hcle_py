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

    def _will_step(self):
        if not self._in_game: self.skip_between_rounds()

    def _did_step(self):
        # If match has started and no save exists, make one
        self.episode_frame_count += 1

        if self._has_backup: return
        if self._in_game: return
        self._backup()

    def get_current_score(self):
        byte1 = np.uint64(self.current_ram[self.ram_dict["score_first_byte"]])
        byte2 = np.uint64(self.current_ram[self.ram_dict["score_second_byte"]])
        byte3 = np.uint64(self.current_ram[self.ram_dict["score_third_byte"]])

        return byte1 + byte2*0x100 + byte3*0x100*0x100
    
    def get_previous_score(self):
        byte1 = np.uint64(self.previous_ram[self.ram_dict["score_first_byte"]])
        byte2 = np.uint64(self.previous_ram[self.ram_dict["score_second_byte"]])
        byte3 = np.uint64(self.previous_ram[self.ram_dict["score_third_byte"]])

        return byte1 + byte2*0x100 + byte3*0x100*0x100
    
    def get_score_change(self) -> int:
        return self.get_current_score() - self.get_previous_score()
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = -0.1
        reward += float(self.get_score_change())

        print(self.get_current_score())

        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self.current_ram[self.ram_dict["game_over"]] == 0xA
        return False