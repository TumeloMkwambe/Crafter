from torch import nn
from torchvision import models

class Actor_Critic(nn.Module):

    def __init__(self, n_actions):
        super().__init__()

        resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.actor = nn.Linear(resnet.fc.in_features, n_actions)

        self.critic = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):

        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        logits = self.actor(features)
        
        values = self.critic(features)

        return logits, values