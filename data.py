from torch.utils.data import Dataset


class Dataset(Dataset):
    
    def __init__(self, observations, actions, log_probs, values, returns, advantages):
        
        self.observations = observations
        
        self.actions = actions
        
        self.log_probs = log_probs
        
        self.values = values
        
        self.returns = returns
        
        self.advantages = advantages

    def __len__(self):
        
        return len(self.observations)

    def __getitem__(self, idx):
        
        obs = self.observations[idx]
        
        acts = self.actions[idx]
        
        log_probs = self.log_probs[idx]
        
        values = self.values[idx]
        
        returns = self.returns[idx]
        
        advantages = self.advantages[idx]

        return obs, acts, log_probs, values, returns, advantages

