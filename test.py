
from mtl import make_env
from train_low import train_map

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()


PERM=1
env=make_env(1,PERM=PERM)
train_map(env,77,750,PERM)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())