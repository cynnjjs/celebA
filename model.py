from torch import nn
import torch.nn.functional as F

class myConvNet(nn.Module):
    def __init__(self, output_size=2):
        super(myConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1) # in-channel, out-channel, kernel size, stride, padding
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pool =  nn.AvgPool2d(4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 64 -> 32
        x = F.relu(self.conv2(x)) # 32 -> 16
        x = F.relu(self.conv3(x)) # 16 -> 8
        x = self.pool(F.relu(self.conv4(x))) # 8 -> 4 -> 1
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(model_type, input_size, output_size):
    hidden_sizes = [128, 64]

    if model_type == 'Linear':
        # Linear Model
        model = nn.Sequential(nn.Linear(input_size, output_size))
    elif model_type == '3-layer':
        # 3-layer model
        model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size))
    else:
        # ConvNet with input size 64*64 for celebA
        model = myConvNet(output_size=output_size)

    return model
