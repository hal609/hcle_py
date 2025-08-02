import gymnasium as gym
import multiprocessing as mp
import nes_gym
import time
import sdl2

def make_env(game_name:str, envs_create:int=1, framestack:int=4, render_mode:str="rgb_array", fps_limit:int=-1) -> gym.vector.AsyncVectorEnv:
    print(f"Creating {envs_create} envs")

    def create_env(game_name:str, render_mode:str="rgb_array"):
        env = (gym.make(f'NES/{game_name}-v1', render_mode=render_mode, max_episode_steps=10000))

        # return gym.wrappers.FrameStackObservation(env, stack_size=framestack)
        return gym.wrappers.FrameStack(env, framestack)
    
    return gym.vector.AsyncVectorEnv(
        [lambda: create_env(game_name, render_mode=render_mode) for _ in range(envs_create)],
        context="spawn",  # Required for Windows
    )

def main():
      game = "SuperMarioBros3"
      print("Currently Playing Game: " + str(game))

      env = make_env(game, 1, framestack=4, render_mode="human")
      print(env.observation_space)
      print(env.action_space[0])

      steps = 0
      observation, info = env.reset()

      while True:
         steps += 1
         start = time.time()
         env.step_async([0])
         observation_, reward, done_, trun_, info = env.step_wait()
         time.sleep(max(1/60 - (time.time() - start), 0))
         observation = observation_

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()