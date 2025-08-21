"""GymNESium environment for SMB2 (International)"""

import numpy as np
from ..nes_env import *

# Expanded inputs to allow for more nuanced control
inputs = [
    NES_INPUT_NONE,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_A,  # Jump
    NES_INPUT_B,  # Run / Pick up
    NES_INPUT_DOWN | NES_INPUT_B, # Crouch / Power Jump
    NES_INPUT_RIGHT | NES_INPUT_A,
    NES_INPUT_RIGHT | NES_INPUT_B,
    NES_INPUT_LEFT | NES_INPUT_A,
    NES_INPUT_LEFT | NES_INPUT_B,
]

# --- RAM Address Constants ---
GAME_STATE = 0x00CD          # 0 on title/menus, non-zero in game
PLAYER_X_PAGE = 0x0014       # Player's horizontal screen/page
PLAYER_X_POS = 0x0028        # Player's X position on the current screen
PLAYER_HEALTH = 0x04C2       # Player's current health (0x0F, 0x1F, etc.)
LIVES = 0x04ED               # Number of lives
LEVEL_TRANSITION = 0x04EC    # 02=game over, 03=end level
CURRENT_LEVEL = 0x0531       # Current level ID (00=1-1, etc.)
CURRENT_AREA = 0x04E7        # Current area within the level

class SMB2Env(NESEnv):
    """An environment for playing Super Mario Bros. 2 with Gymnasium."""

    def __init__(self, render_mode:str = "rgb_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Initialize a new SMB2 environment.
        """
        super().__init__("smb2", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)
        self.setActions(inputs)

    @property
    def _in_game(self) -> bool:
        """Return True if the player is in a playable level."""
        return self.current_ram[GAME_STATE] != 0

    def skip_between_rounds(self) -> None:
        """If on the title or character select screen, press Start/A to advance."""
        while not self._in_game:
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_A)
            self._frame_advance(0)

    def _get_progress_score(self, ram: np.array) -> int:
        """
        Calculates a composite score representing game progress.
        This handles screen wraps and non-linear level progression.
        """
        level = int(ram[CURRENT_LEVEL])
        area = int(ram[CURRENT_AREA])
        x_page = int(ram[PLAYER_X_PAGE])
        x_pos = int(ram[PLAYER_X_POS])

        # Combine values into a single large number.
        # Level is most important, then area, then horizontal position.
        # The multipliers ensure that a small change in a higher-order
        # value (like changing area) is always greater than a large
        # change in a lower-order value (like x_pos).
        return (level * 100000) + (area * 10000) + (x_page * 256) + x_pos

    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        # 1. Calculate progress based on the composite score
        current_progress = self._get_progress_score(self.current_ram)
        previous_progress = self._get_progress_score(self.previous_ram)
        progress_reward = float(current_progress - previous_progress)

        # 2. Small time penalty to encourage speed
        time_penalty = -0.0

        # 3. Penalty for taking damage
        damage_penalty = 0.0
        if self.current_ram[PLAYER_HEALTH] < self.previous_ram[PLAYER_HEALTH]:
            damage_penalty = -25.0

        # 4. Large reward for finishing a level
        level_finish_bonus = 0.0
        if self.current_ram[LEVEL_TRANSITION] == 0x03:
            level_finish_bonus = 100.0

        # Only reward positive progress to avoid penalizing necessary leftward movement
        # within a new area. The large bonus from changing areas will outweigh this.
        reward = max(progress_reward, 0) + time_penalty + damage_penalty + level_finish_bonus
        if reward != 0: print(reward)
        return reward

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        # Episode ends on Game Over or if lives decrease
        game_over = self.current_ram[LEVEL_TRANSITION] == 0x02
        lost_life = self.current_ram[LIVES] < self.previous_ram[LIVES]
        return game_over or lost_life