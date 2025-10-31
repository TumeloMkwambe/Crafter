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
        
        info['achievements']['return'] = reward

        achievements = {key: achievements[key] + info['achievements'][key] if key in achievements else info['achievements'][key] for key in info['achievements']}

        if log:

            wandb.log(achievements)
    
    if log:
        
        wandb.finish()
    
    achievements['episode_length'] = episode_length

    return achievements

def metrics(env, model, n_episodes = 10, log = False, run_name = None):

    achievement_rates = {}

    for _ in range(n_episodes):

        achievements = evaluation_episode(env, model, log, run_name)

        rewards = achievements['return']
        episode_length = achievements['episode_length']
        
        achievement_masks = {key: 1 if achievements[key] != 0 else 0 for key in achievements}

        achievement_masks['return'] = rewards
        achievement_masks['episode_length'] = episode_length
        
        for key, value in achievement_masks.items():

            if key not in achievement_rates:

                achievement_rates[key] = 0

            achievement_rates[key] += value
    
    for key in achievement_rates:

        achievement_rates[key] /= n_episodes

    nontrivial_keys = [key for key in achievement_rates if key not in ['return', 'episode_length']]

    percents = np.array([achievement_rates[key] * 100 for key in nontrivial_keys])

    crafter_score = float(np.exp(np.mean(np.log(1 + percents))) - 1)

    geometric_mean = crafter_score / 100

    achievement_rates['geometric_mean'] = geometric_mean
    achievement_rates['crafter_score_percents'] = crafter_score

    return achievement_rates
