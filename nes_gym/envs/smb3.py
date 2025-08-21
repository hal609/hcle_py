"""GymNESium environment for Super Mario Bros. 3"""

from ..nes_env import *
import random

inputs = [
    NES_INPUT_NONE,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,                           # Move right
    NES_INPUT_RIGHT | NES_INPUT_B,             # Run right
    NES_INPUT_RIGHT | NES_INPUT_A,             # Jump right
    NES_INPUT_RIGHT | NES_INPUT_B | NES_INPUT_A, # Running jump right
    NES_INPUT_A,                               # Jump in place
]

class SMB3Env(NESEnv):
    """An environment for playing Super Mario Bros. 3 with OpenAI Gym."""

    def __init__(self, render_mode:str = "rgb_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Initialize a new Super Mario Bros. 3 environment.

        Args:
            render_mode (str): Optional - "rgb_array" or "human". Defines if a window is displayed.
            fps_limit (int): The frame rate limit of the game. Negative values are unlimited. Defaults to -1.
            max_episode_steps (int): The maximum number of steps per episode. Negative values are unlimited. Defaults to -1.
        """
        # Make sure you have a ROM named 'smb3.nes' accessible to the environment loader.
        super().__init__("smb3", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

        # RAM addresses for Super Mario Bros. 3 based on the provided map
        self.ram_dict = {
            "x_pos_hi": 0x0075,       # Horizontal position in pages (units of 8 blocks)
            "x_pos_lo": 0x0090,       # Horizontal position within the page
            "player_state": 0x00ED,   # 0=Small, 1=Super, 2=Fire, 3=Raccoon, etc.
            "lives": 0x0736,        # Mario's remaining lives
            "in_level_timer": 0x05EE, # Level timer (upper byte). Non-zero when in a level.
            "p_meter": 0x03DD,        # P-meter in status bar (0x7F = full)
            "win_flag": 0x066F,       # Card selection at end of level. Non-zero on win.
            "is_on_map": 0x0014,      # Flag to return to map screen. 0 when in a level.
            "map_screen_y": 0x7976,
            "map_screen_x": 0x797A,
            "world_num": 0x0727,
            "hit_timer": 0x0552,
            "is_dying": 0x00F1,
            "title_state": 0x00DE,
        }

    @property
    def map_x(self) -> int:
        return self.nes[self.ram_dict["map_screen_x"]]
    
    @property
    def map_y(self) -> int:
        return self.nes[self.ram_dict["map_screen_y"]]
    
    @property
    def _in_game(self) -> bool:
        """Return True if the agent is currently in a playable level."""
        # A reliable check is to see if the "return to map" flag is 0
        # and the level timer is running.
        # print("Map screen X:", self.nes[self.ram_dict["map_screen_x"]], "Map screen Y:", self.nes[self.ram_dict["map_screen_y"]])
        # print("Hit timer", self.nes[self.ram_dict["hit_timer"]])
        # print("Title state", self.nes[self.ram_dict["title_state"]])
        return self.nes[self.ram_dict["is_on_map"]] == 0 and self.nes[self.ram_dict["in_level_timer"]] > 0

    def advance_n_frames(self, n:int, action:int = 0) -> None:
        for i in range(n):        
            self._frame_advance(action)
            if self._in_game: break

    def skip_between_rounds(self) -> None:
        """If the agent is on the map screen, spam START to enter a level."""
        while not self._in_game: 
            if self.nes[self.ram_dict["title_state"]] != 0:
                self._frame_advance(0)
                self._frame_advance(NES_INPUT_START)

            elif self.nes[self.ram_dict["world_num"]] == 0:
                if self.map_x == 32 and self.map_y == 64:
                    self.advance_n_frames(240)
                    self._frame_advance(NES_INPUT_RIGHT)
                    self.advance_n_frames(60)
                    self._frame_advance(NES_INPUT_UP)
                    self.advance_n_frames(60)
                    self._frame_advance(NES_INPUT_A)
                    self._frame_advance(0)
                elif (self.map_x == 64 or self.map_x == 128) and self.map_y == 32:
                    self._frame_advance(0)
                    self._frame_advance(NES_INPUT_RIGHT)
                    self._frame_advance(0)
                    self._frame_advance(NES_INPUT_A)
                elif self.map_x == 160 and self.map_y == 32:
                    self.advance_n_frames(240*4)
                    self._frame_advance(NES_INPUT_RIGHT)
                    self.advance_n_frames(120)
                    self._frame_advance(NES_INPUT_DOWN)
                    self.advance_n_frames(120)
                    self._frame_advance(NES_INPUT_LEFT)
                    self.advance_n_frames(120)
                    self._frame_advance(NES_INPUT_A)
                    self._frame_advance(0)
                elif self.map_x == 160 and self.map_y == 64:
                    self.advance_n_frames(240)
                    self._frame_advance(NES_INPUT_LEFT)
                    self.advance_n_frames(60)
                    self._frame_advance(NES_INPUT_DOWN)
                    self.advance_n_frames(60)
                    self._frame_advance(NES_INPUT_LEFT)
                    self.advance_n_frames(60)
                    self._frame_advance(NES_INPUT_A)
                    self._frame_advance(0)
                else:
                    self._frame_advance(0)
            else:
                # Spam random buttons to advance from map to level
                # self._frame_advance(random.randint(0, 255))
                self._frame_advance(0)

            if self.get_done():
                return
    
    def get_mario_pos(self) -> int:
        """Get Mario's absolute horizontal position."""
        return np.int64(self.current_ram[self.ram_dict["x_pos_hi"]]) * 256 + np.int64(self.current_ram[self.ram_dict["x_pos_lo"]])

    def get_mario_pre_pos(self) -> int:
        """Get Mario's horizontal position from the previous frame."""
        return np.int64(self.previous_ram[self.ram_dict["x_pos_hi"]]) * 256 + np.int64(self.previous_ram[self.ram_dict["x_pos_lo"]])

    def get_pos_change(self) -> int:
        """Get the change in Mario's horizontal position."""
        return int(self.get_mario_pos()) - int(self.get_mario_pre_pos())

    def get_reward(self) -> float:
        """Return the reward for the current step."""
        # Time penalty to encourage finishing the level faster
        reward = -0.1

        # Reward for moving to the right
        x_pos_change = float(self.get_pos_change())
        # Clamp reward to prevent large values from screen transitions or glitches
        if x_pos_change > 10 or x_pos_change < -10:
            x_pos_change = 0
        reward += x_pos_change

        # Check for death
        if self.current_ram[self.ram_dict["lives"]] < self.previous_ram[self.ram_dict["lives"]] or (self.nes[self.ram_dict["is_dying"]] != 0):
            reward -= 20.0

        # Check for winning by seeing if the end-of-level card roulette is active
        if self.current_ram[self.ram_dict["win_flag"]] > 0:
            reward += 50.0  # Large reward for winning

        # Small reward for increasing the P-meter to encourage running
        if self.current_ram[self.ram_dict["p_meter"]] > self.previous_ram[self.ram_dict["p_meter"]]:
            reward += 0.5

        return reward

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        # Episode is over if Mario dies or wins the level
        is_dead = self.current_ram[self.ram_dict["lives"]] < self.previous_ram[self.ram_dict["lives"]] or (self.nes[self.ram_dict["is_dying"]] != 0)
        return is_dead