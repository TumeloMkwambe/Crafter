import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch

from config import config

def observation_preprocessing(observations):
        
    observations = torch.tensor(observations).unsqueeze(0)
    observations = observations.permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W]
    observations = observations.float() / 255.0
        
    return observations

def evaluation_episode(env, model, log = False, run_name = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log:

        wandb.init(entity = "reinforcement-learning-wits", project = "Crafter", name = run_name, config = config)
    
    observation, info = env.reset()
    
    terminated = False

    achievements = {}

    episode_length = 0

    while not terminated:

        episode_length += 1

        observation = observation_preprocessing(observation)
        
        model.eval()
        
        with torch.no_grad():

            observation = observation.to(device)
            
            logits, _ = model(observation)
            
            distros = torch.distributions.Categorical(logits = logits)
            
            action = int(distros.sample())
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        info['achievements']['reward'] = reward

        achievements = {key: achievements[key] + info['achievements'][key] if key in achievements else info['achievements'][key] for key in info['achievements']}

        if log:

            wandb.log(achievements)
    
    if log:
        
        wandb.finish()
    
    achievements['episode_length'] = episode_length

    return achievements

def metrics(env, model, n_episodes):

    achievement_rates = {}

    for _ in range(n_episodes):

        achievements = evaluation_episode(env, model)

        rewards = achievements['reward']
        episode_length = achievements['episode_length']
        
        achievement_masks = {key: 1 if achievements[key] != 0 else 0 for key in achievements}

        achievement_masks['reward'] = rewards
        achievement_masks['episode_length'] = episode_length
        
        for key, value in achievement_masks.items():

            if key not in achievement_rates:

                achievement_rates[key] = 0

            achievement_rates[key] += value
    
    for key in achievement_rates:

        achievement_rates[key] /= n_episodes
    
    nontrivial_keys = [k for k in achievement_rates if k not in ["reward", "episode_length"]]
    eps = 1e-8
    values = [max(achievement_rates[k], eps) for k in nontrivial_keys]

    geometric_mean = float(np.exp(np.mean(np.log(values)))) if len(values) > 0 else 0.0
    achievement_rates["geometric_mean"] = geometric_mean

    return achievement_rates
