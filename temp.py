import json

f1 = open("camera_params.test", "r")
data_RC = json.load(f1)['params']


f2 = open("images.test", "r")
data_P = json.load(f2)['data']

def valid(l):
    for x in l:
        if x != 0:
            return True
    return False

data_P = [x for x in data_P if x['height'] != 0 and valid(x['P']) ]
files = [x['name'] for x in data_P]

# Camera details
P = np.array([np.array(x['P']).reshape([3, 4]) for x in data_P])
R = np.array([np.array(x['R']).reshape([3, 3]) for x in data_RC])
C = np.array([np.array(x['C']) for x in data_RC])
K = np.matmul(P[:, :, :3], np.linalg.inv(R))
