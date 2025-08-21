"""GymNESium environment for Kung Fu"""

from ..nes_env import *

inputs = [
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_B,
    NES_INPUT_A,
    ]

GAME_STATE = 0x0006 # 0x in game 0x moving direction 0x58 on title
ATTRACT = 0x06B # Set to 1 when in attract mode, otherwise 0
IN_PLAY = 0x0390 # 1 when playing
HP = 0x04A6 # Begins at 0x30
ACTIONABLE = 0x60 # 0 When actionable
MENU = 0x5C # Set to 0 on menu
X_FINE = 0xD4
X_LARGE = 0xA3
FLOOR = 0x58
DEAD = 0x038D # Set to 0xFF when dead, otherwise 0

class KungFuEnv(NESEnv):
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
        super().__init__("kungfu", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

    @property
    def in_menu(self) :
        '''Return the current game mode.'''
        return self.current_ram[MENU] == 0x0
    
    def _will_step(self):
        while (self.in_menu):
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
        # while self.current_ram[IN_PLAY] != 1:  self._frame_advance(0)


    def _did_step(self) -> None:
        # If the level has just started and no backup state exists, create one.
        if not self._has_backup and (not self.in_menu) and self.current_ram[IN_PLAY]:
            self._backup()

    
    @property
    def score_change(self) -> int:
        return self.score() - self.score(ram=self.previous_ram)
    
    def score(self, ram=None) -> int:
        if ram is None: ram = self.current_ram
        score:np.int64 = 0
        mult = 1
        for addr in [0x0535, 0x0534, 0x0533, 0x0532]:
            score += np.int64(ram[addr]) * mult
            mult *= 10
        return score

    @property
    def x_change(self) -> np.int64:
        change = np.int64(self.current_ram[X_FINE]) - np.int64(self.previous_ram[X_FINE])
        if abs(change) == 255: return -change/255
        elif abs(change) > 3: return 0
        return change
    
    @property
    def hp_change(self) -> np.int64:
        change = np.int64(self.current_ram[HP]) - np.int64(self.previous_ram[HP])
        if change > 0 or np.int64(self.current_ram[HP]) == 0: return 0
        return change

    @property
    def dead(self) -> bool:
        return self.current_ram[DEAD] != 0
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        if self.current_ram[ATTRACT] == 1: return 0

        reward = -0.01

        if self.current_ram[FLOOR] % 2 == 0:
            reward += -self.x_change
        else:
            reward += self.x_change
        reward += self.score_change/100
        reward += self.hp_change
        if self.dead: reward -= 100

        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self.dead