import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        
        
        self.S1=60
        self.S2=30
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Linear(4, self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S1, self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S2,2),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S2, self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S1,4)
        )

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        lr=1e-2
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def decode(self,x):
        x = self.decoder(x)
        return x

    def encode(self,x):
        x = self.encoder(x)
        return x


    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.encode(x)
        return pred.detach().numpy()
        
    
    def train(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.forward(x)
        loss=self.loss_fn(pred,x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    def save(self,PATH):
        torch.save(self.state_dict(), PATH)

    def load(self,PATH):
        self.load_state_dict(torch.load(PATH))