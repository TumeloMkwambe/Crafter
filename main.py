import stable_baselines3 as sb3

from evaluation import evaluation_episode
from actor_critic import Actor_Critic
from env_setup import init_env
from config import config
from ppo import PPO

if __name__ == "__main__":

    env_vec = sb3.common.env_util.make_vec_env(env_id = init_env, n_envs = 5)

    ac = Actor_Critic(n_actions = 10)
    ppo = PPO(vec_env = env_vec, model = ac, config = config)

    ppo.data_collection()

    ppo.stack()

    ppo.advantages_collector()

    ppo.flatten()

    ac = ppo.training_loop()

    env = init_env()

    evaluation_episode(env, ppo.model, config, "base-1")