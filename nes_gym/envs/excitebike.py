"""GymNESium environment for Excitebike"""

from ..nes_env import *

inputs = [
    NES_INPUT_NONE,
    NES_INPUT_LEFT,
    NES_INPUT_RIGHT,
    NES_INPUT_A,       # Accelerate (Normal)
    NES_INPUT_B,       # Accelerate (Turbo)
    NES_INPUT_A | NES_INPUT_LEFT,
    NES_INPUT_A | NES_INPUT_RIGHT,
    NES_INPUT_B | NES_INPUT_LEFT,
    NES_INPUT_B | NES_INPUT_RIGHT,
]

# --- RAM Address Constants ---
RACING_FLAG = 0x004F      # 00 = At start, 01 = Racing
PLAYER_SPEED = 0x00F3     # Player's current speed (0-47)
MOTOR_TEMP = 0x03E3       # Bike's heat level (overheats at 32)
GAME_TIMER_MIN = 0x0068   # Race timer minutes
GAME_TIMER_SEC = 0x0069   # Race timer seconds
GAME_TIMER_HUN = 0x006A   # Race timer hundredths of a second
PLAYER_STATUS = 0x00F2    # Status of the bike (e.g., falling)

class ExcitebikeEnv(NESEnv):
    """An environment for playing Excitebike with Gymnasium."""

    def __init__(self, render_mode:str = "rgb_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Initialize a new Excitebike environment.
        """
        super().__init__("excitebike", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)
        
        self.setActions(inputs)
        self.finish_time = None

    @property
    def _in_game(self) -> bool:
        """Return True if the player is currently racing."""
        return self.current_ram[RACING_FLAG] == 0x01

    def _did_reset(self):
        self.finish_time = None
        if not self._has_backup:
            for i in range(30):
                self._frame_advance(NES_INPUT_NONE)
                self._frame_advance(NES_INPUT_START)
        return super()._did_reset()

    def skip_between_rounds(self) -> None:
        """If the agent is not in a race, press A to start."""
        # In Excitebike, holding A or B gets through the menus and starts the race.
        while not self._in_game:
            self._frame_advance(NES_INPUT_A)

    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        # Small penalty for existing to encourage finishing quickly
        reward = -0.01

        # Reward based on current speed
        speed = self.current_ram[PLAYER_SPEED]
        reward += float(speed) / 10.0 # Scale speed to a reasonable reward value

        # Heavy penalty for overheating
        if self.current_ram[MOTOR_TEMP] >= 32:
            reward -= 20.0

        # Penalty for falling (indicated by a non-zero status)
        if self.current_ram[PLAYER_STATUS] != 0 and self.current_ram[PLAYER_STATUS] != 4:
            reward -= 5.0

        reward = reward / 1e4
        return reward

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        # If the race is finished, the timer stops updating. We can detect this.
        current_time = self.read_mult_byte([GAME_TIMER_HUN, GAME_TIMER_SEC, GAME_TIMER_MIN])

        if self._in_game and self.finish_time is None:
            # Check if the timer has stopped. A simple way is to see if the
            # current time is the same as the previous frame's time.
            previous_time = self.read_mult_byte([GAME_TIMER_HUN, GAME_TIMER_SEC, GAME_TIMER_MIN], ram_selection=self.previous_ram)
            if current_time == previous_time and current_time > 0:
                self.finish_time = current_time
                return True

        return False