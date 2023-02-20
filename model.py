import copy

import timm
import torch
import torch.nn as nn

def create_model(num_classes=7, drop_rate=0):
    model = ResNet18(num_classes=num_classes, drop_rate=drop_rate)

    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0):
        super(ResNet18, self).__init__()
        self.drop_rate = drop_rate

        model = timm.create_model('resnet18', pretrained=False)
        checkpoint = torch.load('./pretrain/resnet18_msceleb.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.fc = nn.Linear(512, num_classes)

        self.feature = nn.Sequential(*list(model.children())[:-5])
        for param in self.feature.parameters():
            param.requires_grad = False

        self.branch_1 = nn.Sequential(*list(model.children())[-5:])
        self.branch_1_feature = nn.Sequential(*list(self.branch_1.children())[:-1])
        self.branch_1_classifier = self.branch_1[-1]
        
        self.branch_2 = copy.deepcopy(self.branch_1)
        self.branch_2_feature = nn.Sequential(*list(self.branch_2.children())[:-1])
        self.branch_2_classifier = self.branch_2[-1]
        
        self.alpha_1 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())    
        self.alpha_2 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())  

    def forward(self, image):
        feature_0 = self.feature(image)
        feature_0 = nn.Dropout(self.drop_rate)(feature_0)

        feature_1 = self.branch_1_feature(feature_0)
        feature_2 = self.branch_2_feature(feature_0)

        attention_weights_1 = self.alpha_1(feature_1)
        attention_weights_2 = self.alpha_2(feature_2)

        out_1 = attention_weights_1 * self.branch_1_classifier(feature_1)
        out_2 = attention_weights_2 * self.branch_2_classifier(feature_2)

        attention_weights = (attention_weights_1 + attention_weights_2) / 2
        
        return out_1, out_2, attention_weights