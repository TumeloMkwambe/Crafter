import os
import torch
import stable_baselines3 as sb3

from evaluation import metrics
from actor_critic import Actor_Critic_I, Actor_Critic_II
from env_setup import init_env
from config import config
from ppo import PPO

def train_ppo_agent(agent_name):

    env_vec = sb3.common.env_util.make_vec_env(env_id = init_env, n_envs = config['NUM_ENVS'])

    model = Actor_Critic_I(n_actions = env_vec.action_space.n)

    if os.path.exists(os.path.join('models', agent_name)):

        state_dict = torch.load(os.path.join('models', agent_name), map_location = 'cpu')
        model.load_state_dict(state_dict)
    
    for i in range(10):

        ppo = PPO(vec_env = env_vec, actor_critic = model, config = config, divergence_check = True)

        ppo.learn(total_timesteps = config['TOTAL_STEPS'], verbose = False)

        env_vec.close()

        env = init_env()

        agent_metrics = metrics(env = env, model = ppo.actor_critic, n_episodes = 10, log = False)

        model = ppo.actor_critic.cpu()

        print(f'RUN {i + 1} | AGENT METRICS: {agent_metrics}\n')

        env.close()

        torch.save(model.state_dict(), os.path.join('models', agent_name))

def report_metrics(agent, agent_name):

    for run in range(10):

        log = False

        if run == 9:

            log = True

        env = init_env()

        agent_metrics = metrics(env = env, model = agent, n_episodes = 10, log = log, run_name = agent_name)

        env.close()

        print(f'{agent_name} METRICS: {agent_metrics}\n')

        with open(f'metric-reports/{agent_name}-metrics.txt', 'a') as file:

            file.write(f'RUN {run + 1}:\n')

            for key, value in agent_metrics.items():
                
                file.write(f"{key}: {value}\n")
            
            file.write('\n')

if __name__ == "__main__":

    agent_name = 'ppo-agent-i' # ppo-agent-i (resnet, no kl check), ppo-agent-ii (naturecnn, kl check), ppo-agent-iii (naturecnn, no kl check)

    #train_ppo_agent(agent_name) Used to train the agent and save the model

    env = init_env()

    model = Actor_Critic_II(n_actions = env.action_space.n) # change to Actor_Critic_I for ppo-agent-i, else Actor_Critic_II for ppo-agent-ii and ppo-agent-iii
    state_dict = torch.load(f'./models/{agent_name}', map_location = 'cpu')
    model.load_state_dict(state_dict)

    env.close()

    report_metrics(agent = model, agent_name = agent_name)
