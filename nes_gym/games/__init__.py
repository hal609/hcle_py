# nes_gym/games/__init__.py

from .smb1 import SMB1Env
from .mtpo import MTPOEnv
# from .tetris import TetrisEnv

# The central registry for all supported NES games
GAME_REGISTRY = {
    "SuperMarioBros": {
        "logic_class": SMB1Env,
    },
    "MikeTysonsPunchOut": {
        "logic_class": MTPOEnv,
    },
    # "Tetris": {
    #     "logic_class": TetrisEnv,
    # },
}