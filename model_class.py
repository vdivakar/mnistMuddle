import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        #Encoder
        self.conv1 = nn.Conv2d( 1,  8, 3, stride=2,padding=1) #28x28x1 -> 14x14x8
        self.conv2 = nn.Conv2d( 8, 16, 3, stride=2,padding=1) #14x14x8 -> 7x7x16
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2) #7x7x16 -> 3x3x32
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1) #3x3x32 -> 1x1x32
        self.fc1   = nn.Linear(32, 10)
        
        #Decoder
        self.t_fc1   = nn.Linear(10, 32) # [32]
        self.t_conv1 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16,  8, 3, stride=2, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(8,   1, 3, stride=2, output_padding=1)
    
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x))        
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return x
    
    def decoder(self, x):
        x = F.relu(self.t_fc1(x)) # 10 -> 32
        x = x.reshape([-1,32,1,1])
        x = F.relu(self.t_conv1(x)) # [m, 1,1,32] -> [m, 3,3,32]
        x = F.relu(self.t_conv2(x)) # [m, 3,3,32] -> [m, 7,7,16]
        x = F.relu(self.t_conv3(x)) # [m, 7,7,16] -> 
        x = F.relu(self.t_conv4(x))
        return x                    #[m, 1, 28, 28]
    
    def forward(self, x):
        latent_vector    = self.encoder(x)
        predicted_output = self.decoder(latent_vector)
        return predicted_output
        
