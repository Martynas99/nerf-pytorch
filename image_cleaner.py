from unicodedata import name
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import visdom
























def main():
    split = True
    viz = visdom.Visdom()
    max_sum = 0
    for i in range(120):
        path = f"logs/tea/renderonly_path_049999/{i:03}_acc.png"
        image = cv2.imread(os.path.join("/home/mifs/ml867/test_nerf/temp/nerf-pytorch", path),0)
        if np.sum(image) > max_sum:
            i_max = i
            max_sum = np.sum(image)
    print(i_max)
    path = f"logs/tea/renderonly_path_049999/{i_max:03}.png"
    image = cv2.imread(os.path.join("/home/mifs/ml867/test_nerf/temp/nerf-pytorch", path))
    kernel = np.ones((18,18),np.uint8)
    if split and image.shape[-1]==3:
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing2opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    opening2closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    if split and image.shape[-1]==3:
        closing = np.moveaxis(closing, [2,0,1], [0,1,2])
        opening = np.moveaxis(opening, [2,0,1], [0,1,2])
        image = np.moveaxis(image, [2,0,1], [0,1,2])
        viz.image(opening, opts={'title': 'Opening'})
        viz.image(closing, opts={'title': 'Closing'})
        viz.image(image, opts={'title': 'Original'})
        
    else:
        viz.image(opening, opts={'title': 'Opening'})
        viz.image(closing, opts={'title': 'Closing'})
        viz.image(opening2closing, opts={'title': 'Opening + Closing'})
        viz.image(closing2opening, opts={'title': 'Closing + Opening'})
        viz.image(image, opts={'title': 'Original'})

    




if __name__ == "__main__":
    main()