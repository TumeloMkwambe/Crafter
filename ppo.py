import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import config
from data import Dataset

class PPO:

    def __init__(self, vec_env, model, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vec_env = vec_env
        self.model = model.to(self.device)
        self.n_steps = config['N_STEPS']
        self.gamma = config['GAMMA']
        self.lambda_ = config['LAMBDA']

        self.data = {"observations": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": [], "infos": []}
        self.count_steps = 0

    def observation_preprocessing(self, observations):
        
        observations = torch.tensor(observations)
        observations = observations.permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W]
        observations = observations.float() / 255.0
        
        return observations

    def data_collection(self):
        
        observations = self.vec_env.reset()
        
        while self.count_steps < self.n_steps:

            observations = self.observation_preprocessing(observations)

            self.model.eval()
            
            with torch.no_grad():
                
                logits, values = self.model(observations)
                
                distros = torch.distributions.Categorical(logits = logits)
                
                actions = distros.sample().unsqueeze(-1)
                
                log_probs = distros.log_prob(actions.squeeze(-1)).unsqueeze(-1)
                
            next_observations, rewards, dones, infos = self.vec_env.step(actions.cpu().numpy().squeeze(-1))
            
            self.data["observations"].append(observations) 
            self.data["actions"].append(actions)
            self.data["log_probs"].append(log_probs)
            self.data["values"].append(values)
            self.data["rewards"].append(rewards)
            self.data["dones"].append(dones)
            self.data["infos"].append(infos)
            
            self.count_steps += self.vec_env.num_envs
            observations = next_observations

        with torch.no_grad():

            observations = self.observation_preprocessing(observations)

            _, values = self.model(observations)

            self.data["values"].append(values)

    def stack(self):

        obs = self.data["observations"][0].shape

        self.data["observations"] = torch.stack(self.data["observations"]).to(self.device)
        
        self.data["actions"] = torch.cat(self.data["actions"]).view(len(self.data["actions"]), -1).to(self.device)

        self.data["log_probs"] = torch.cat(self.data["log_probs"]).view(len(self.data["log_probs"]), -1).to(self.device)

        self.data["values"] = torch.cat(self.data["values"]).view(len(self.data["values"]), -1).to(self.device)

        self.data["rewards"] = torch.tensor(np.stack(self.data["rewards"]), dtype = torch.float32).to(self.device)
        
        self.data["dones"] = torch.tensor(np.stack(self.data["dones"]), dtype = torch.float32).to(self.device)
    
    def advantages_collector(self):

        rewards = self.data['rewards']

        values = self.data["values"]

        dones = self.data["dones"]

        T, n_envs = rewards.shape
        
        gae = torch.zeros(n_envs, device = self.device)
        
        advantages = torch.zeros_like(rewards, device = self.device)
        
        for t in reversed(range(T)):
            
            delta = rewards[t] + config['GAMMA'] * values[t + 1] * (1 - dones[t]) - values[t]
            
            gae = delta + config['GAMMA'] * config['LAMBDA'] * (1 - dones[t]) * gae
            
            advantages[t] = gae

        returns = advantages + values[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.data['returns'] = returns
        self.data['advantages'] = advantages

    def flatten(self):

        self.data["observations"] = torch.flatten(self.data["observations"], start_dim = 0, end_dim = 1)

        self.data["actions"] = torch.flatten(self.data["actions"], start_dim = 0, end_dim = 1)

        self.data["log_probs"] = torch.flatten(self.data["log_probs"], start_dim = 0, end_dim = 1)

        self.data["advantages"] = torch.flatten(self.data["advantages"], start_dim = 0, end_dim = 1)

        self.data["returns"] = torch.flatten(self.data["returns"], start_dim = 0, end_dim = 1)

    def training_loop(self):

        dataset = Dataset(
            self.data["observations"],
            self.data["actions"],
            self.data["log_probs"],
            self.data["returns"],
            self.data["advantages"]
        )
        
        dataloader = DataLoader(dataset, batch_size = config['BATCH_SIZE'], shuffle = True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr = config['LEARNING_RATE'])
        mse = nn.MSELoss()

        self.model.train()
        
        for epoch in range(config['N_EPOCHS']):

            print(f'EPOCH {epoch + 1}')
        
            clip_loss = 0.0
            vf_loss = 0.0
            entropy_bonus = 0.0
            train_loss = 0.0
        
            for obs, acts, log_probs, returns, advantages in dataloader:

                obs = obs.to(self.device)
                acts = acts.to(self.device)
                log_probs = log_probs.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)

                optimizer.zero_grad()
                
                new_logits, new_values = self.model(obs)

                new_distros = torch.distributions.Categorical(logits = new_logits)
                
                new_log_probs = new_distros.log_prob(acts.squeeze(-1))
        
                ratios = torch.exp(new_log_probs - log_probs)
        
                unclipped = ratios * advantages
        
                clipped = torch.clamp(ratios, 1 - config['EPSILON'], 1 + config['EPSILON']) * advantages
        
                L_CLIP = -torch.mean(torch.min(clipped, unclipped))
        
                L_VF = mse(new_values, returns.view(-1, 1))
        
                entropy = new_distros.entropy().mean()
        
                loss = L_CLIP + config['C1'] * L_VF - config['C2'] * entropy

                clip_loss += L_CLIP.item()
                vf_loss += L_VF.item()
                entropy_bonus += entropy.item()
                train_loss += loss.item()
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['MAX_GRAD_NORM'])
                optimizer.step()
        
            clip_loss /= len(dataloader)
            vf_loss /= len(dataloader)
            entropy_bonus /= len(dataloader)
            train_loss /= len(dataloader)

            print(f'Loss: {train_loss} | L_CLIP: {clip_loss} | L_VF: {vf_loss} | Entropy: {entropy_bonus}\n')
