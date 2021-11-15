import json
import numpy as np
from PIL import Image
import os
import cv2
    




def pos_from_data(path, id_list):
    c2w, u, v = [], [], []
    K = np.load(os.path.join(path,'Ks/K_{:03d}.npy'.format(id_list[0])))
    W = 2 * K[0][2]
    for id in id_list:
        image = Image.open(os.path.join(path, '{:03d}.png'.format(id)))
        image = np.asarray(image)[:,:,:3]
        print(image.shape)
        print(np.where((image == (255,0,0)).all(axis=2)))
        # FINDS v,u where pixel is red 
        y,x = np.where((image == (255,0,0)).all(axis=2))
        print(y,x)
        c2w.append(np.load(os.path.join(path, 'c2ws/c2w_{:03d}.npy'.format(id)))) 
        u.append(x)
        v.append(y)
    #CHECK IF PRODUCED POINT IS AS EXPECTED
    out_vec = pos_from_pixel(K, c2w, u, v)
    c2w = c2w[0]
    c2w_arr = np.linalg.inv(np.vstack((c2w[:3,:4], np.array([0,0,0,1]))))
    pixel_pos = K @ c2w_arr[:3,:4] @ out_vec.T
    print(pixel_pos/pixel_pos[-1])
    return out_vec


def pos_from_pixel(K, c2w, u, v):
    arr = np.zeros((2*len(u), 4))
    for i in range(len(u)):
        # Two equations per image (for u and v)
        c2w_i = c2w[i]
        c2w_i = np.vstack((c2w_i[:3,:4], np.array([0,0,0,1])))
        R_T = np.linalg.inv(c2w_i)
        P = K @ R_T[:3,:4]
        arr[2*i][0] = P[0][0] - u[i]*P[2][0]
        arr[2*i][1] = P[0][1] - u[i]*P[2][1]
        arr[2*i][2] = P[0][2] - u[i]*P[2][2]
        arr[2*i][3] = P[0][3] - P[2][3]
        arr[2*i+1][0] = P[1][0] - v[i]*P[2][0]
        arr[2*i+1][1] = P[1][1] - v[i]*P[2][1]
        arr[2*i+1][2] = P[1][2] - v[i]*P[2][2]
        arr[2*i+1][3] = P[1][3] - P[2][3]
    # Using A.T @ A because optimal solution is 0 otherwise
    out =  np.linalg.eig(arr.T @ arr)
    # Find vec with minimal eiganvalue and rescale to form (X,Y,Z,1)
    return out[1][np.argmin(out[0])] / out[1][np.argmin(out[0])][-1]
        
def main():
    path = "./logs/tea/annotated"
    # path = "./logs/tea/renderonly_path_049999"
    l = np.array([[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3],[2,2,3]]])
    # print(l.shape)
    
    point_pos = pos_from_data(path, [0, 30])
    print(point_pos)
    

if __name__=="__main__":
    main()