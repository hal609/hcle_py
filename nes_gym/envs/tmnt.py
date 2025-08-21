"""GymNESium environment for Teenage Mutant Ninja Turtles"""

from ..nes_env import *
import numpy as np

# A good set of inputs for TMNT
inputs = [
    NES_INPUT_NONE,
    NES_INPUT_RIGHT,
    NES_INPUT_LEFT,
    NES_INPUT_UP,
    NES_INPUT_DOWN,
    NES_INPUT_A,         # Jump
    NES_INPUT_B,         # Attack
    NES_INPUT_A | NES_INPUT_RIGHT,
    NES_INPUT_A | NES_INPUT_LEFT,
]

# --- RAM Address Constants ---
OVERWORLD_X = 0x0010
OVERWORLD_Y = 0x0011
LEVEL_X = 0x00A0
MAP_ID = 0x0020
BOSS_HEALTH = 0x04D0
TURTLE_HEALTH_START = 0x0077
LIVES = 0x0046
IN_GAME = 0x003C
ON_CSS = 0x0035

class TMNTEnv(NESEnv):
    """An environment for playing Teenage Mutant Ninja Turtles."""

    def __init__(self, render_mode:str = "rgb_array", **kwargs):
        super().__init__("tmnt", render_mode=render_mode, **kwargs)
        self.setActions(inputs)
        # --- State for tracking exploration ---
        self.visited_overworld_coords = set()

    def on_reset(self):
        """Reset exploration tracker on new episode."""
        self.visited_overworld_coords.clear()

    @property
    def _in_game(self) -> bool:
        return self.current_ram[IN_GAME] == 1
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in game then spam start until the next level begins.'''
        while (not self._in_game):
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
    
    def _did_step(self) -> None:
        while self.current_ram[ON_CSS] == 1:
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)

        return super()._did_step()
    @property
    def _is_on_overworld(self):
        """Check if the player is on the main map."""
        # Map ID for the overworld is 0x00
        return self.current_ram[MAP_ID] == 0x00

    def get_reward(self) -> float:
        reward = -0.01  # 1. Time penalty to encourage speed

        # 2. Exploration and Progress Rewards
        if self._is_on_overworld:
            coords = (self.current_ram[OVERWORLD_X], self.current_ram[OVERWORLD_Y])
            if coords not in self.visited_overworld_coords:
                reward += 0.5  # Reward for discovering a new part of the map
                self.visited_overworld_coords.add(coords)
        else: # Inside a level
            x_progress = int(self.current_ram[LEVEL_X]) - int(self.previous_ram[LEVEL_X])
            reward += max(x_progress, 0) # Reward forward movement, ignore backtracking

        # 3. Key Event Rewards and Penalties
        # Check for boss defeat
        if self.previous_ram[BOSS_HEALTH] > 0 and self.current_ram[BOSS_HEALTH] == 0:
            reward += 50.0 # Large reward for defeating a boss

        # Check for taking damage (any turtle)
        for i in range(4):
            health_addr = TURTLE_HEALTH_START + i
            if self.current_ram[health_addr] < self.previous_ram[health_addr]:
                if self.current_ram[health_addr] == 0:
                    reward -= 30.0 # Larger penalty for dying
                else:
                    reward -= 5.0 # Penalty for getting hit
        if reward != -0.01: print(reward)
        return reward / 100.0 # Scale reward to a reasonable range

    def get_done(self) -> bool:
        """Episode is over if lives are depleted."""
        return self.current_ram[LIVES] != 3