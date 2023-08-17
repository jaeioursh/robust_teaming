import torch
import numpy as np

class Net:
    def __init__(self):
        learning_rate=1e-2
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30,1)
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)

    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()


net=Net()

x=np.random.random((4,8))
y=np.random.random((4,1))

net.feed(x)
for i in range(100):

    print(net.train(x,y))
