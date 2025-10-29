import matplotlib.pyplot as plt
import wandb
import torch
import cv2

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

def image_display(img):
    
    plt.figure(figsize = (6, 6))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.axis('off')
    
    plt.show()