import os
import time
import numpy as np
import copy
from .cynes.__init__ import *
from .cynes.__init__ import NES
from .cynes.windowed import WindowedNES
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from abc import abstractmethod

# Define NES input constants
NES_INPUT_NONE = 0x00
NES_INPUT_RIGHT = 0x01
NES_INPUT_LEFT = 0x02
NES_INPUT_DOWN = 0x04
NES_INPUT_UP = 0x08
NES_INPUT_START = 0x10
NES_INPUT_SELECT = 0x20
NES_INPUT_B = 0x40
NES_INPUT_A = 0x80

INPUTS = [NES_INPUT_NONE,NES_INPUT_RIGHT,NES_INPUT_LEFT,NES_INPUT_DOWN,NES_INPUT_UP,NES_INPUT_START,NES_INPUT_SELECT,NES_INPUT_B,NES_INPUT_A]

class NESEnv(gym.Env):
    ''' NES Gymnasium Environment. '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, game_name:str, render_mode:str = "rbg_array", fps_limit:int = -1, max_episode_steps:int = -1) -> None:
        """
        Create a new NES environment.

        Args:
            rom_path (str): The path to the NES .rom file to be loaded.
            render_mode (str): Optional - Either "rgb_array" or "human"  which defines whether the environment should display a window for each env.
            fps_limit (int): The frame rate limit of the game, negative values are unlimited. Defaults to -1

        Returns:
            None

        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._has_backup = False # Initially no state has been saved
        self.done = True # Setup a done flag

        current_dir = os.path.dirname(os.path.abspath(__file__))
        rom_path = os.path.join(current_dir, 'roms', f'{game_name}.bin')

        # Create either windowless or windowed instance of the cynes emulator
        if render_mode == "rgb_array":
            self.nes = NES(rom_path)
        elif render_mode == "human":
            self.nes = WindowedNES(rom_path)
        else:
            raise Exception("Invalid render mode passed. Valid render modes are 'rgb_array' and 'human'.")

        # Define observation and action spaces
        self.observation_space = Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        self.action_space = Discrete(256)

        self.fps_limit = fps_limit
        self.reward_range = (-float(np.inf), float(np.inf))

        self.setActions()

        self.last_time = time.time()

        self.episode_frame_count = 0
        self.max_episode_length = max_episode_steps

        self.current_ram = np.zeros(2048)
        self.previous_ram = np.zeros(2048)

    def setActions(self, actionList: list = INPUTS):
        self._action_map = actionList
        self.action_space = gym.spaces.Discrete(len(self._action_map))
    
    def value_change(self, address: int) -> int:
        '''Return the difference between a RAM value at the current frame and the previous frame.'''
        return int(int(self.current_ram[address]) - int(self.previous_ram[address]))
    
    def reset(self, seed=None, options=None):
        '''
        Reset the emulator to the last save, or to power on if no save is present.

        Args:
            seed (optional int):    The seed that is used to initialize the parent Gym environment's PRNG (np_random).
            
            options (optional dict): Additional information to specify how the environment is reset (optional)
        '''
        super().reset(seed=seed, options=options)

        # Call the before reset callback
        self._will_reset()

        # Reset the emulator
        if self._has_backup: self._restore()
        else: self.nes.reset()

        self.episode_frame_count = 0
        
        # Call the after reset callback
        self._did_reset()

        self.done = False
        obs = self.nes.step(frames=1)  # Capture an initial frame
        obs = np.array(obs, dtype=np.uint8)  # Ensure it's a valid numpy array
        info = {}

        self.current_ram = self.nes.get_all_ram()
        self.previous_ram = copy.deepcopy(self.current_ram)

        return obs, info

    def _backup(self) -> None:
        """Backup the current emulator state."""
        self._backup_state = self.nes.save()
        self._has_backup = True

    def _restore(self) -> None:
        """Restore the emulator state from backup."""
        self.nes.load(self._backup_state)

    def _frame_advance(self, action):
        """
        Advance the emulator by one frame with the given action.

        Args:
            action (int): The action to perform (controller input).
        """
        # Set the controller inputs
        self.nes.controller = action

        # Advance the emulator by one frame
        frame = self.nes.step(frames=1)

        # Update the current frame (observation)
        self.screen = frame

    @abstractmethod
    def _will_reset(self):
        ''' Called just before a reset, can be used to apply any RAM hacking before resetting. '''
        pass
    
    @abstractmethod
    def _did_reset(self):
        ''' Called after a reset, can be used to apply any RAM hacking required after resetting. '''
        pass
    
    def _get_info(self) -> dict:
        """Return the info after a step occurs. Required for gymnasium but not needed here."""
        return {}
    
    @abstractmethod
    def _will_step(self):
        ''' Called just before a reset, can be used to apply any RAM hacking before resetting. '''
        pass
    
    @abstractmethod
    def _did_step(self):
        ''' Called after a reset, can be used to apply any RAM hacking required after resetting. '''
        pass

    @abstractmethod
    def get_reward(self) -> float:
        raise Exception("Reward method must be overridden for each game environment.")

    def max_len_exceeded(self) -> bool:
        if self.max_episode_length < 0: return False
        return self.episode_frame_count > self.max_episode_length
    
    def get_done(self) -> bool:
        return False

    def step(self, action: int) -> tuple:
        ''' Transition function: advances one frame of gameplay with a given action. '''

        if self.done: raise ValueError('Cannot step in a done environment! Call `reset` first.')
        self._will_step()

        self.nes.controller = self._action_map[action]
        frame = self.nes.step(frames=1)
        obs = np.array(frame, dtype=np.uint8)
        reward = float(self.get_reward())
        self.done = bool(self.get_done() or self.max_len_exceeded())

        self.previous_ram = copy.deepcopy(self.current_ram)
        self.current_ram = self.nes.get_all_ram()

        self.episode_frame_count += 1

        # Bound the reward in [min, max]
        if reward < self.reward_range[0]: reward = self.reward_range[0]
        elif reward > self.reward_range[1]: reward = self.reward_range[1]
        
        if self.fps_limit > 0:
            # Sleep until the required frame duration has passed for consistent frame rate
            if (1/self.fps_limit - (time.time() - self.last_time)) > 0:
                time.sleep(1/self.fps_limit - (time.time() - self.last_time))
            self.last_time = time.time()

        self._did_step()

        return obs, reward, self.done, False, {}
    
    def read_mult_byte(self, locations:list, endian:str = "big", ram_selection:np.array = None) -> int:
        if ram_selection is None: ram_selection = self.current_ram
        
        result: int = 0
        mult = 1

        if endian not in ["little", "big"]: raise Exception(f"Attempted multi-byte read with invalid endian argument of: '{endian}'. Valid options are 'little' and 'big'.")
        locations = sorted(locations) if endian == "little" else  sorted(locations, reverse=True)

        for addr in locations:
            result += mult * np.uint64(ram_selection[addr])
            mult *= 0x100

        return result