"""GymNESium environment for NES Baseball"""

from ..nes_env import *

inputs = [
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_B,
    NES_INPUT_A,
    ]

IN_GAME = 0x001E # 1 in game, 0 on title
GAME_STATE = 0x03D0 # 0 in menu, 0x80 in team select and 0x14 in game
ACTIONABLE = 0x0078
IN_CUTSCENE = 0x087 # 1 when locked in, 0 when able to interact
BATTING = 0xF
STRIKES = 0x62
BALLS = 0x63
OUTS = 0x64
SCORE1 = 0x67
SCORE2 = 0x68
IS_TEAM_2 = 0x4B # If equal to 1 then look at score at 0x68

class BaseballEnv(NESEnv):
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
        super().__init__("baseball", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

    @property
    def in_menu(self) :
        '''Return the current game mode.'''
        return self.current_ram[IN_GAME] == 0x0
    
    @property
    def batting(self) :
        '''Return the current game mode.'''
        return self.current_ram[BATTING] == 0x1
    
    @property
    def bases(self):
        return max((int(f"{self.current_ram[0x38d]:08b}"[4:], 2)-10)/2, 0)
    
    def _will_step(self):
        print(self.bases)
        while (self.in_menu):
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
        while self.current_ram[GAME_STATE] == 0x80:
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_A)


    def _did_step(self) -> None:
        # If the level has just started and no backup state exists, create one.
        if not self._has_backup and (not self.in_menu) and (self.current_ram[GAME_STATE] != 0x80):
            self._backup()

    @property
    def score_change(self) -> int:
        return self.score() - self.score(ram=self.previous_ram)
    
    @property
    def opp_score_change(self) -> int:
        return self.opp_score() - self.opp_score(ram=self.previous_ram)
    
    def score(self, ram=None) -> np.int64:
        if ram is None: ram = self.current_ram
        if ram[IS_TEAM_2]:
            return np.int64(ram[SCORE2])
        else:
            return np.int64(ram[SCORE1])
        
    def opp_score(self, ram=None) -> np.int64:
        if ram is None: ram = self.current_ram
        if ram[IS_TEAM_2]:
            return np.int64(ram[SCORE1])
        else:
            return np.int64(ram[SCORE2])
    
    @property
    def score_change(self) -> np.int64:
        return self.score(ram=self.current_ram) - self.score(ram=self.previous_ram)
    
    @property
    def balls_change(self) -> np.int64:
        delta = np.int64(self.current_ram[BALLS]) - np.int64(self.previous_ram[BALLS])
        if delta != 1: return 0
        return delta
    
    @property
    def outs_change(self) -> np.int64:
        delta = np.int64(self.current_ram[STRIKES]) - np.int64(self.previous_ram[STRIKES])
        if delta != 1: return 0
        return delta
    
    @property
    def strikes_change(self) -> np.int64:
        delta = np.int64(self.current_ram[OUTS]) - np.int64(self.previous_ram[OUTS])
        if delta != 1: return 0
        return delta
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = 0
        reward += self.score_change * 500
        reward -= self.opp_score_change * 500
        if self.batting:
            reward += self.bases * 100
            reward -= self.balls_change
            reward -= self.outs_change * 10
            reward -= self.strikes_change * 100
        else:
            reward += self.balls_change
            reward += self.outs_change * 10
            reward += self.strikes_change * 100
        reward /= 1e2
        # if reward != 0: print(reward)
        return reward

        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return False