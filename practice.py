import numpy as np
import pickle

Reward_Data=np.arra
Calc= np.zeros((sample_size,2))
count=0
sample_size = 100
 while index < sample_size:
        test_index=np.random.randint(0,sample_size-1)
        row_index=np.random.randint(0,sample_size-1)
        G1 = Reward_Data[row_index,0]
        G2 = Reward_Data[test_index,0]
        
        while G1==G2:
            test_index=np.random.randint(0,sample_size-1)
            G2 = Reward_Data[test_index,0]
            if [G1,G2] in Calc:     
                count=sample_size
                break
