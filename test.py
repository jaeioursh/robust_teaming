
from mtl import make_env

env=make_env(1,2,0,1,1)
print(env.worldTrainStepFuncCol)

S, r, done, info = env.step([[1,1]])
S=S[0]
print(S)
