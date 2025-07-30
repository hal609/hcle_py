"""GymNESium environment for Mike Tyson's Punch Out!!"""

from ..nes_env import *

# inputs = [NES_INPUT_NONE,NES_INPUT_RIGHT,NES_INPUT_LEFT,NES_INPUT_DOWN,NES_INPUT_UP,NES_INPUT_START,NES_INPUT_SELECT,NES_INPUT_B,NES_INPUT_A]
inputs = [NES_INPUT_NONE, NES_INPUT_LEFT, NES_INPUT_B, NES_INPUT_UP | NES_INPUT_B]

class MTPOEnv(NESEnv):
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
        super().__init__("mtpo", render_mode=render_mode, fps_limit=fps_limit, max_episode_steps=max_episode_steps)

        self.setActions(inputs)

        self.ram_dict = {
            "opp_id": 0x0001,
            "mac_hp": 0x0391,
            "opp_hp": 0x0398,
            "opp_ko_count": 0x03D1,
            "timer_mins_digit": 0x0302,
            "timer_tens_digit": 0x0304,
            "timer_digit": 0x0305
        }

    @property
    def _in_fight(self) :
        '''Return the current round number.'''
        return self.current_ram[0x0004] == 0xFF
    
    def skip_between_rounds(self) -> None:
        ''' If agent is not in fight then spam start until the next round begins.'''
        while (not self._in_fight):
            self._frame_advance(0)
            self._frame_advance(0)
            self._frame_advance(NES_INPUT_START)
            self._frame_advance(NES_INPUT_START)

    def _will_step(self):
        if not self._in_fight: self.skip_between_rounds()


    def _did_step(self):
        # If match has started and no save exists, make one
        if self.current_ram[self.ram_dict["timer_digit"]] != 0 and not self._has_backup:
            self._backup()
    
    def get_time_dif(self) -> float:
        return 60*self.value_change("timer_mins_digit") + 10*self.value_change("timer_tens_digit") + self.value_change("timer_digit")

    def get_reward(self) -> float:
        """Return the reward after a step occurs."""
        hit_reward = max(-self.value_change("opp_hp"), 0)
        health_penalty = max(-self.value_change("mac_hp"), 0)
        return hit_reward - health_penalty
        next_opp_reward = self.value_change("opp_id")
        ko_reward = self.value_change("opp_ko_count")
        health_penalty = max(-self.value_change("mac_hp"), 0)
        time = self.get_time_dif()
        reward = (200* next_opp_reward) + (2* ko_reward) +  hit_reward -  health_penalty
        # if reward != 0: print("Reward:", reward)
        return (200* next_opp_reward) + (2* ko_reward) +  hit_reward -  health_penalty

    def get_done(self) -> bool:
        """Return True if the episode is over, False otherwise."""
        # if self.value_change("mac_hp") != 0: print("mac hp", self.value_change("mac_hp"))
        return self.value_change("mac_hp") < -3