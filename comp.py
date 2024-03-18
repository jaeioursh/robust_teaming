import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import time;
def scaled_dot_product_attention(query, key, value) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L,D = query.size(-2), key.size(-1)
    scale_factor = 1 / math.sqrt(D)
    attn_bias = torch.zeros(L, L, dtype=query.dtype)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value,attn_weight

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P



class net(nn.Module):
    def __init__(self,dim=[],hidden=20):
        super(net, self).__init__()
        self.d=dim

        self.w1=nn.Linear(dim[0],dim[1])
        
        self.w2=nn.Linear(dim[1],dim[2])
        

        self.rnn = nn.LSTM(dim[2], dim[3], 1,batch_first=True)
        self.c=torch.randn(1,dim[3])
        self.h=torch.randn(1,dim[3])
        #self.rnn = nn.RNN(dim[2], dim[3], 1,batch_first=True)

        self.w3=nn.Linear(dim[3],dim[4])

        self.w4=nn.Linear(dim[4],dim[5])

        self.w5=nn.Linear(dim[5],dim[6])

        self.ww=nn.Linear(dim[2],dim[4])

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        lr=1e-2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.acti=nn.LeakyReLU()
        #self.acti=nn.Sigmoid()

    def forward(self,x):
        
        x=self.acti(self.w1(x))
        x=self.acti(self.w2(x))
        x,_ = self.rnn(x)
        x=self.acti(self.w3(x))
        x=self.acti(self.w4(x))
        x=self.w5(x)
        return x

    def memory(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        x=self.acti(self.w1(x))
        x=self.acti(self.w2(x))
        x,mem = self.rnn(x)
        return mem[0].detach().numpy(),mem[1].detach().numpy()
    
    def gen(self,x,d1,d2):
        self.h[:]=0
        self.c[0,0]=d1
        self.c[0,1]=d2
        x=torch.from_numpy(x.astype(np.float32))
        x=self.acti(self.w1(x))
        x=self.acti(self.w2(x))
        x,_ = self.rnn(x,(self.h,self.c))
        x=self.acti(self.w3(x))
        x=self.acti(self.w4(x))
        x=self.w5(x)
        return x.detach().numpy()
    
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.forward(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.forward(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()
    



x=[]
y=[]
np.random.seed(0)
for i in range(500):
    t1=np.random.rand()+1
    t2=np.random.rand()+2
    t=np.linspace(0,5,20)
    sig=np.sin(t*t1)+np.sin(t*t2)
    zer=np.zeros_like(sig)
    sig,zer=np.reshape(sig,(-1,1)),np.reshape(zer,(-1,1))
    x.append(np.vstack([sig,zer]))
    y.append(np.vstack([zer,sig]))

#x,y=[x[0]],[y[0]]
X,Y=np.array(x,dtype=float),np.array(y,dtype=float)
print(X.shape)

#    0 1   2 3 4  5  6
dim=[1,100,100,2,100,100,1]
atn=net(dim)

if 1:
    for i in range(3001):
        #for x,y in zip(X,Y):
        l=atn.train(X,Y)
        print(i,l)

        #if i%100==0:
        #    print(np.round(atn.feed(X[0]),3))
    torch.save(atn.state_dict(), "tests/A")
else:
    atn.load_state_dict(torch.load("tests/A"))
    h,c=atn.memory(X[:,:20,:])
    h,c=h[0],c[0]
    print(h.shape)
    fig1=plt.figure(1)
    #plt.scatter(h[:,0],h[:,1])
    plt.scatter(c[:,0],c[:,1])
    fig2=plt.figure(2)
    fig2.clear()
    f1,=plt.plot(X[0,:20,:])
    f2,=plt.plot(X[0,:20,:])
    tt=time()
    def onclick(event):
        global tt
        if time()-tt>0.1:
            tt=time()
            d1,d2=[event.xdata,event.ydata]
            d=c-np.array([d1,d2])
            d=d[:,0]**2.0+d[:,1]**2.0
            idx=np.argmin(d)
            print(idx)
            
            y=atn.gen(X[idx,20:,:],d1,d2)
            f1.set_ydata(X[idx,:20,:])
            f2.set_ydata(y[:])
            fig2.canvas.draw()
            fig2.canvas.flush_events()

        #plt.show()
    evt='motion_notify_event'
    #evt='button_press_event'
    fig1.canvas.mpl_connect(evt, onclick)
    
    plt.show()