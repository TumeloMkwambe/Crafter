import numpy as np

if not hasattr(np, "bool8"):

    np.bool8 = np.bool_


from shimmy import GymV21CompatibilityV0
import gym
import crafter

def init_env():
    
    gym.envs.registration.register(id = 'CrafterNoReward-v1', entry_point = crafter.Env)
    
    env = gym.make('CrafterNoReward-v1')

    if not hasattr(env, "seed"):

        env.seed = lambda seed = None: None
    
    env = crafter.Recorder(
        env, './path/to/logdir',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    
    env = GymV21CompatibilityV0(env = env)
    
    return env 