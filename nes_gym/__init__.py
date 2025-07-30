"""
NES-GYM: A Gymnasium environment for Nintendo Entertainment System games.
"""
# This file makes nes_gym a package and runs registration on import.
from .registration import register_nes_envs

# Call the function automatically when the package is imported
register_nes_envs()