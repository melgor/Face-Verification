 
In [1]: import scipy.io as s

In [2]: import yaml

In [3]: d = dict()

In [4]: d['1'] = mat['Model3D'][0,0][1]
KeyboardInterrupt

In [4]: mat = s.loadmat('model3D_DLIB.mat')

In [5]: d['0'] = mat['Model3D'][0,0][0].tolist()

In [6]: d['1'] = mat['Model3D'][0,0][1].tolist()

In [7]: d['2'] = mat['Model3D'][0,0][2].tolist()

In [8]: d['3'] = mat['Model3D'][0,0][3].tolist()

In [9]: d['4'] = mat['Model3D'][0,0][4].tolist()

In [10]: d['5'] = mat['Model3D'][0,0][5].tolist()

In [11]: d['6'] = mat['Model3D'][0,0][6].tolist()

In [12]: d['7'] = mat['Model3D'][0,0][7].tolist()

In [13]: with open('dlibModel3D.yaml', 'w') as f:
            yaml.dump(d,f)
