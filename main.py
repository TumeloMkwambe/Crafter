import os
import torch
import wandb
import stable_baselines3 as sb3

from evaluation import metrics
from actor_critic import Actor_Critic
from env_setup import init_env
from config import config
from ppo import PPO

def train_ppo_agent():

    env_vec = sb3.common.env_util.make_vec_env(env_id = init_env, n_envs = config['NUM_ENVS'])

    model = Actor_Critic(n_actions = env_vec.action_space.n)

    if os.path.exists('ppo-agent-ii'):

        state_dict = torch.load('ppo-agent-ii', map_location = 'cpu')
        model.load_state_dict(state_dict)
    
    for i in range(10):

        ppo = PPO(vec_env = env_vec, actor_critic = model, config = config)

        ppo.learn(total_timesteps = config['TOTAL_STEPS'], verbose = False)

        env_vec.close()

        env = init_env()

        log = False

        if i == 9:
            log = True

        agent_metrics = metrics(env = env, model = ppo.actor_critic, n_episodes = 10, log = log)

        model = ppo.actor_critic.cpu()

        print(f'RUN {i + 1} | AGENT METRICS: {agent_metrics}\n')

        env.close()
        
        torch.save(model.state_dict(), 'ppo-agent-ii')

if __name__ == "__main__":

    #train_ppo_agent()

    env = init_env()

    model = Actor_Critic(n_actions = env.action_space.n)

    state_dict = torch.load('ppo-agent-ii', map_location = 'cpu')
    model.load_state_dict(state_dict)

    agent_metrics = metrics(env = env, model = model, n_episodes = 10, log = True)
    
    env.close()

    print(f'PPO AGENT I AGENT METRICS: {agent_metrics}\n')