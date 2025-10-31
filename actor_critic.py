import torch
from torch import nn
from torchvision import models

class Actor_Critic_I(nn.Module):

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

class Actor_Critic_II(nn.Module): # First Improvement: Fine-tune Pretrained ResNet50 As Feature Extractor.

    def __init__(self, n_actions):
        super().__init__()

        resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.actor = nn.Linear(resnet.fc.in_features, n_actions)

        self.critic = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):

        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)

        logits = self.actor(features)
        
        values = self.critic(features)

        return logits, values