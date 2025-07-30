# nes_gym/registration.py

import gymnasium as gym

GAME_REGISTRY = {
    "SuperMarioBros": "nes_gym.games.smb1:SMB1Env",
    "MikeTysonsPunchOut": "nes_gym.games.mtpo:MTPOEnv",
    "Tetris": "nes_gym.games.tetris:TetrisEnv"
}

def register_nes_envs():
    """Registers all games in the GAME_REGISTRY with Gymnasium."""
    for game_name, entry_point in GAME_REGISTRY.items():
        env_id = f"NES/{game_name}-v1"

        if env_id in gym.envs.registry:
            continue

        gym.register(
            id=env_id,
            entry_point=entry_point, # Gymnasium handles the import
            nondeterministic=True,
        )