import gymnasium as gym
import nes_gym

print("Creating single environment for visualization...")

def make_env(envs_create:int=1, framestack:int=4, render_mode:str="rgb_array", fps_limit:int=-1) -> gym.vector.AsyncVectorEnv:
    '''
    Create a vectorised game environment.

    Args:
        envs_create (int): The number of parallel environments to create. Defaults to 1
        framestack (int): The number of frames which are stacked together to form 1 observation. Defaults to 4
        headless (bool): Whether the environments should be headless, i.e. no window is displayed. Defaults to False
        fps_limit (int): Integer limit for the fps of the environment. Negative values give unlimited fps. Defaults to -1

    Returns:
        gym.vector.AsyncVectorEnv: Vectorised Gym environment.
    '''
    print(f"Creating {envs_create} envs")

    def create_env():
        env = gym.make('NES/Tetris-v1', render_mode="human")

        return gym.wrappers.FrameStackObservation(env, stack_size=framestack)
    
    return gym.vector.AsyncVectorEnv(
        [lambda: create_env() for _ in range(envs_create)],
        context="spawn",  # Required for Windows
    )

if __name__ == "__main__":
    env = make_env(1, framestack=4)

    obs, info = env.reset()
    done = False
    while True:
        action = env.action_space.sample() # Using random action for demo

        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    print("Evaluation finished.")