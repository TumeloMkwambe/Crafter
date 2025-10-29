from torch import nn
from torchvision import models

class Actor_Critic(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 8, stride = 4, padding = 0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.actor = nn.Linear(1024, n_actions)

        self.critic = nn.Linear(1024, 1)

    def forward(self, x):

        features = self.cnn(x)

        logits = self.actor(features)
        
        values = self.critic(features)

        return logits, values