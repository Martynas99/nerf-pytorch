import json
import numpy as np
from PIL import Image
import os

    




def pos_from_data(path, id_list):
    c2w, K, u, v = [], [], [], []

    for id in id_list:
        image = Image.open(os.path.join(path, '{:03d}.png'.format(id)))
        image = np.asarray(image)
        print(image.shape)
        x,y = np.where((image == (255,0,0)).all(axis=2))
        K = np.load(os.path.join('K_{:03d}.npy'.format(id)))
        c2w = np.load(os.path.join('c2w_{:03d}.npy'.format(id)))
        u.append(x)
        y.append(W-y)
        




def pos_from_pixel(K, c2w, u, v):
    arr = np.zeros((2*len(u), 4))
    u.extend(v)
    for i in range(2*len(u)):
        P = K[i] @ np.linalg.inv(c2w)[:3][:4]
        z = int(i/len(u))
        arr[i][0] = P[z][0] - u[i]*P[2][0]
        arr[i][1] = P[z][1] - u[i]*P[2][1]
        arr[i][2] = P[z][2] - u[i]*P[2][2]
        arr[i][3] = P[z][3] - u[i]*P[2][3]
    return np.linalg.lstsq(arr)
        
def main():
    path = "./logs/tea/annotated"
    l = np.array([[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[2,2,3]]])
    # print(l.shape)
    
    # point_pos = pos_from_data("./logs/tea/renderonly_path_049999", [0])

    a = np.load(os.path.join(path, "Ks/K_023.npy"))
    print(a)

if __name__=="__main__":
    main()