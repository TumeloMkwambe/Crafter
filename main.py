import os
import torch
import wandb
import stable_baselines3 as sb3

from evaluation import evaluation_episode
from actor_critic import Actor_Critic
from env_setup import init_env
from config import config
from ppo import PPO

def train_ppo_agent():

    env_vec = sb3.common.env_util.make_vec_env(env_id = init_env, n_envs = config['NUM_ENVS'])

    model = Actor_Critic(n_actions = env_vec.action_space.n)

    if os.path.exists('ppo-agent'):

        state_dict = torch.load('ppo-agent', map_location = 'cpu')
        model.load_state_dict(state_dict)

    for i in range(10):

        ppo = PPO(vec_env = env_vec, model = model, config = config)

        ppo.data_collection()

        ppo.stack()

        ppo.advantages_collector()

        ppo.flatten()

        ppo.training_loop(verbose = False)

        env_vec.close()

        env = init_env()

        achievements = evaluation_episode(env = env, model = ppo.model, log = False)

        model = ppo.model

        print(f'RUN {i + 1} | ACHIEVEMENTS: {achievements}\n')

        env.close()
    
    torch.save(model.state_dict(), 'ppo-agent')

if __name__ == "__main__":

    train_ppo_agent()