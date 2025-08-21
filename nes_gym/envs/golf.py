"""GymNESium environment for NES Golf"""

from ..nes_env import *

inputs = [
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_A
    ]

SCORE = 0x00F
GAME_STATE = 0x0002 # 0x83 in game 0xC3 moving direction 0x63 on title
ON_GREEN = 0x0049 # 1 when on green 0 otherwise
STROKES = 0x002C
BALL_Y = 0x0040
CAN_SHOOT = 0x0008 # 1 when can shoot 0 when ball rolling or on menu
DIST_TO_HOLE = [0x0058, 0x0059]
DIST_ON_GREEN = 0x5E #0x6B
PAR = 0x83
HOLE_NUM = 0x2B

class GolfEnv(NESEnv):
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
        super().__init__("golf", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

    @property
    def _in_game(self) :
        '''Return the current game mode.'''
        return self.current_ram[GAME_STATE] != 0x63
        # return self.current_ram[CAN_SHOOT] == 1
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        while (not self._in_game):
            self._frame_advance(0)
            self._frame_advance(0)
            # if self.current_ram[GAME_STATE] == 0x63:
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(NES_INPUT_START)

    def dist_to_hole(self, ram=None) -> float:
        if ram is None: ram = self.current_ram
        raw_dist = self.read_mult_byte(DIST_TO_HOLE, endian="little", ram_selection=ram)
        if ram[ON_GREEN] == 1: raw_dist = ram[DIST_ON_GREEN]

        return float(raw_dist)
    
    @property
    def strokes_change(self) -> int:
        return int(self.current_ram[STROKES] != self.previous_ram[STROKES])
    
    @property
    def dist_change(self) -> float:
        return self.dist_to_hole(ram=self.current_ram) - self.dist_to_hole(ram=self.previous_ram)
    
    @property
    def score_change(self) -> np.int64:
        return np.int64(self.current_ram[SCORE]) - np.int64(self.previous_ram[SCORE])
    
    @property
    def hole_change(self) -> np.int64:
        return np.int64(self.current_ram[HOLE_NUM]) - np.int64(self.previous_ram[HOLE_NUM])
    
    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        reward = 0.01
        reward -= min(self.dist_change, 0)
            # print(dis_change, (dis_change + (1/dis_change))*(-(np.e**-dis_change)+0.99999))
            # reward += np.e**(-min(self.dist_change, 0) - 1) - 1
        reward -= self.strokes_change * 1000
        reward -= max(self.score_change, 0) * 10
        reward += 10000 * self.hole_change

        reward /= 1e4

        return reward
        
    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        return self.current_ram[STROKES] > self.current_ram[PAR]*2