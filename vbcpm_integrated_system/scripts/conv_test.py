import scipy.signal as ss
import numpy as np

a = [[0,0,0,0,0],
     [0,1,1,0,0],
     [0,0,1,1,0],
     [0,0,0,0,0],
     [0,0,0,0,0]
    ]
b = [[1,0],
     [1,1]]
a = np.array(a)
b = np.array(b)
c = ss.correlate(a,b,mode='valid')
print(c)

xs, ys = b.nonzero()
cx, cy = (c==0).nonzero()
for i in range(len(cx)):
    print('transformed point sum: ')
    print(a[xs + cx[i], ys+cy[i]].sum())
