import torch
import wandb
import stable_baselines3 as sb3

from evaluation import evaluation_episode
from actor_critic import Actor_Critic
from env_setup import init_env
from config import config
from ppo import PPO

if __name__ == "__main__":

    env_vec = sb3.common.env_util.make_vec_env(env_id = init_env, n_envs = config['NUM_ENVS'])

    model = Actor_Critic(n_actions = env_vec.action_space.n)

    #wandb.init(entity = "reinforcement-learning-wits", project = "Crafter", name = "base", config = config)

    for i in range(10):

        ppo = PPO(vec_env = env_vec, model = model, config = config)

        ppo.data_collection()

        ppo.stack()

        ppo.advantages_collector()

        ppo.flatten()

        ppo.training_loop(verbose = False)

        env_vec.close()

        env = init_env()

        achievements = evaluation_episode(env, ppo.model)

        model = ppo.model

        #wandb.log(achievements)

        print(f'RUN {i + 1} | ACHIEVEMENTS: {achievements}\n')

        env.close()
    
    torch.save(model.state_dict(), 'model-base')

    #wandb.finish()
