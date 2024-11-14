import timm
import torch.nn as nn

class network(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super(network, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes = self.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

def load_model(model_name, pretrained, num_classes):
    load_model = network(model_name, pretrained, num_classes)
    return load_model