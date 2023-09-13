import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self,nstate=4):
        super(Autoencoder, self).__init__()

        
        
        self.S1=60
        self.S2=30
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Linear(nstate, self.S1),
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
            nn.Linear(self.S1,nstate)
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


class DIAYN(nn.Module):
    def __init__(self,classes,nstate=8):
        super(DIAYN, self).__init__()

        
        
        self.S1=100
        self.S2=50
        self.model = nn.Sequential( # like the Composition layer you built
            nn.Linear(nstate, self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S1, self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S2,classes),
            #nn.Sigmoid()
            nn.Softmax()
        )
        

        #self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.loss_fn = nn.CrossEntropyLoss()
        lr=1e-2
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.int64))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    
    def save(self,PATH):
        torch.save(self.state_dict(), PATH)

    def load(self,PATH):
        self.load_state_dict(torch.load(PATH))