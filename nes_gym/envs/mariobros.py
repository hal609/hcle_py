"""GymNESium environment for Mario Bros"""

from ..nes_env import *

inputs = [
    NES_INPUT_NONE,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_A,
    NES_INPUT_RIGHT | NES_INPUT_A,
    NES_INPUT_LEFT | NES_INPUT_A,
    ]

P1_SCORE = [0x0095, 0x0096, 0x0097]
P1_LIVES = 0x0048
TIMER = 0x002D

class MarioBrosEnv(NESEnv):
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
        super().__init__("mariobros", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

    @property
    def _in_game(self):
        '''Return the current game mode.'''
        return self.lives > 0
    
    def _will_step(self):
        # Hack timer to 0 to skip wait at start of level
        self.nes[TIMER] = 0
        return super()._will_step()
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        while (not self._in_game):
            self._frame_advance(0)
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(NES_INPUT_START)

    @property
    def lives(self):
        return self.current_ram[P1_LIVES]

    def score_change(self):
        return self.read_mult_byte(P1_SCORE, is_dec=True) - self.read_mult_byte(P1_SCORE, is_dec=True, ram_selection=self.previous_ram)
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = -0.01
        reward += float(self.score_change())
        if  self.lives != 2: reward = -20
        reward = reward / 1e4
        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self.lives == 1