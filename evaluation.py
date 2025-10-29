import matplotlib.pyplot as plt
import wandb
import torch
import cv2

def observation_preprocessing(observations):
        
    observations = torch.tensor(observations).unsqueeze(0)
    observations = observations.permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W]
    observations = observations.float() / 255.0
        
    return observations

def evaluation_episode(env, model, config, run_name = ""):
    
    wandb.init(entity = "reinforcement-learning-wits", project = "Crafter", name = run_name, config = config) 
    
    observation, info = env.reset() 
    
    terminated = False
    
    reward_sum = 0
    
    while not terminated:

        observation = observation_preprocessing(observation)
        
        model.eval()
        
        with torch.no_grad():
            
            logits, _ = model(observation)
            
            distros = torch.distributions.Categorical(logits = logits)
            
            action = int(distros.sample())
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        info['achievements']['reward'] = reward
        
        wandb.log(info['achievements'])
    
    wandb.finish()

def image_display(img):
    
    plt.figure(figsize = (6, 6))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.axis('off')
    
    plt.show()