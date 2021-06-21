from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import os

class Net(nn.Module):
    def __init__(self, arch, n_meta_features: int):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=500, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
        self.meta = nn.Sequential(nn.Linear(n_meta_features, 500),
                                  nn.BatchNorm1d(500),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(500, 250),  # FC layer output will have 250 features
                                  nn.BatchNorm1d(250),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))
        self.ouput = nn.Linear(500 + 250, 1)
        
    def forward(self, inputs):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x, meta = inputs
        cnn_features = self.arch(x)
        meta_features = self.meta(meta)
        features = torch.cat((cnn_features, meta_features), dim=1)
        output = self.ouput(features)
        return output

def load_model():

    script_dir = os.path.dirname(__file__)
    rel_path = "../Data/efficientnet-b1-f1951068.pth"
    MODEL_FILEPATH = os.path.join(script_dir, rel_path)

    arch = EfficientNet.from_pretrained('efficientnet-b1',weights_path=MODEL_FILEPATH)  # Going to use efficientnet-b1 NN architecture
    model = Net(arch=arch, n_meta_features=12)
    model_path = '../Data/model_dict.pth'
    model_path = os.path.join(script_dir, model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model
