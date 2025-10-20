import matplotlib.pyplot as plt
import wandb
import cv2

def evaluation_episode(env, model, config):
    
    wandb.init(entity = "reinforcement-learning-wits", project = "Crafter", name = "", config = config) 
    
    observation, info = env.reset() 
    
    terminated = False
    
    reward_sum = 0
    
    while not terminated:
        
        action, _ = model.predict(observation, deterministic = True)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        info['achievements']['reward'] = reward
        
        wandb.log(info['achievements'])
    
    wandb.finish()


def image_display(img):
    
    plt.figure(figsize = (6, 6))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.axis('off')
    
    plt.show()