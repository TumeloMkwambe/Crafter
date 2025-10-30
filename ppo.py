import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import Dataset

class PPO:

    def __init__(self, vec_env, actor_critic, config):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vec_env = vec_env
        self.actor_critic = actor_critic.to(self.device)
        self.config = config

        self.data = {"observations": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": [], "infos": []}
        self.count_steps = 0
    
    def reset_data(self):
        
        self.data = {"observations": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": [], "infos": []}
        self.count_steps = 0

    def observation_preprocessing(self, observations):
        
        observations = torch.from_numpy(observations).permute(0, 3, 1, 2) # from [N, H, W, C] to [N, C, H, W]
        observations = observations.float() / 255.0
        
        return observations

    def data_collection(self):
        
        observations = self.vec_env.reset()
        
        while self.count_steps < self.config['N_STEPS']:

            observations = self.observation_preprocessing(observations)

            self.actor_critic.eval()
            
            with torch.no_grad():

                observations = observations.to(self.device)
                
                logits, values = self.actor_critic(observations)
                
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
            
            self.count_steps += 1
            observations = next_observations

        with torch.no_grad():

            observations = self.observation_preprocessing(observations)

            observations = observations.to(self.device)

            _, values = self.actor_critic(observations)

            self.data["values"].append(values)

    def stack(self):

        self.data["observations"] = torch.stack(self.data["observations"]).cpu()

        self.data["actions"] = torch.cat(self.data["actions"]).view(len(self.data["actions"]), -1).cpu()

        self.data["log_probs"] = torch.cat(self.data["log_probs"]).view(len(self.data["log_probs"]), -1).cpu()

        self.data["values"] = torch.cat(self.data["values"]).view(len(self.data["values"]), -1).cpu()

        self.data["rewards"] = torch.tensor(np.stack(self.data["rewards"]), dtype = torch.float32).cpu()

        self.data["dones"] = torch.tensor(np.stack(self.data["dones"]), dtype = torch.float32).cpu()

    def advantages_collector(self):

        rewards = self.data['rewards']

        values = self.data["values"]

        dones = self.data["dones"]

        T, n_envs = rewards.shape
        
        gae = torch.zeros(n_envs, device = 'cpu')

        advantages = torch.zeros_like(rewards, device = 'cpu')

        for t in reversed(range(T)):
            
            delta = rewards[t] + self.config['GAMMA'] * values[t + 1] * (1 - dones[t]) - values[t]
            
            gae = delta + self.config['GAMMA'] * self.config['LAMBDA'] * (1 - dones[t]) * gae
            
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

    def training_loop(self, verbose = True):

        dataset = Dataset(
            self.data["observations"],
            self.data["actions"],
            self.data["log_probs"],
            self.data["returns"],
            self.data["advantages"]
        )
        
        dataloader = DataLoader(dataset, batch_size = self.config['BATCH_SIZE'], shuffle = True)

        optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = self.config['LEARNING_RATE'])
        mse = nn.MSELoss()

        self.actor_critic.train()
        
        for epoch in range(self.config['N_EPOCHS']):
        
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

                new_logits, new_values = self.actor_critic(obs)

                new_distros = torch.distributions.Categorical(logits = new_logits)
                
                new_log_probs = new_distros.log_prob(acts.squeeze(-1))
        
                ratios = torch.exp(new_log_probs - log_probs)
        
                unclipped = ratios * advantages
        
                clipped = torch.clamp(ratios, 1 - self.config['EPSILON'], 1 + self.config['EPSILON']) * advantages
        
                L_CLIP = -torch.mean(torch.min(clipped, unclipped))
        
                L_VF = mse(new_values, returns.view(-1, 1))
        
                entropy = new_distros.entropy().mean()
        
                loss = L_CLIP + self.config['C1'] * L_VF - self.config['C2'] * entropy

                clip_loss += L_CLIP.item()
                vf_loss += L_VF.item()
                entropy_bonus += entropy.item()
                train_loss += loss.item()
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['MAX_GRAD_NORM'])
                optimizer.step()
        
            clip_loss /= len(dataloader)
            vf_loss /= len(dataloader)
            entropy_bonus /= len(dataloader)
            train_loss /= len(dataloader)

            if verbose  == True:
                print(f'EPOCH {epoch + 1} | Loss: {train_loss} | L_CLIP: {clip_loss} | L_VF: {vf_loss} | Entropy: {entropy_bonus}\n')

    def learn(self, total_timesteps, verbose = True):

        count_total_steps = 0

        while count_total_steps < total_timesteps:

            self.data_collection()

            count_total_steps += self.config['N_STEPS']

            self.stack()

            self.advantages_collector()

            self.flatten()

            if verbose:

                print(f"= = = = = ITERATION (STEPS: {min(self.count_steps, total_timesteps)} / {total_timesteps}) = = = = =")

            self.training_loop(verbose = False)

            self.reset_data()
    
        return self.actor_critic
