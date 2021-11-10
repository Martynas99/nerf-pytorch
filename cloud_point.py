import matplotlib.pyplot as plt
import numpy as np
import visdom
import pandas as pd
from skspatial.objects import Plane, plane
import pyransac3d as pyrsc
from load_llff import load_llff_data, poses_avg
import logging




def rotation_from_vectors(vec1, vec2):
    v = np.cross(vec1, vec2)
    u = np.dot(vec1, vec2)
    c = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx.dot(vx)*(1-u)/c**2


def find_plane(points):
    plane1 = pyrsc.Plane()
    best_eq, inliers = plane1.fit(points, thresh=0.01, maxIteration=1000)
    plane_normal = np.asarray(best_eq[0:3])
    return best_eq, points[inliers]


def read_points(path):
    points=[]
    with open(path, "r") as f:
        output = f.readlines()
    coords = np.array([i.split(" ")[1:4] for i in output[3:]]).astype(float)
    return coords

def get_rotation_matrix(path, vec):
    points = read_points(path)
    best_eq, plane_points = find_plane(points)
    return rotation_from_vectors(np.asarray(best_eq[0:3]), vec), points

def main():
    basic, rotate = False, True
    recenter, spherify, plot = True, True, True
    bottom = np.reshape([0,0,0,1.], [1,4])

    # Obtain rotation matrix for plane
    rot, points = get_rotation_matrix("./data/nerf_llff_data/tea/points3D.txt", [0,0,-1])
    points = points
    #Pad rotation matrix to get same shape as poses
    pose_rot = np.pad(np.eye(3), ((0,0), (0,2)), mode='constant')
    # 3x5 matrix with first 3x3 being only rotation
    

    # Obtain plane equation and points in plane 
    best_eq, plane_points = find_plane(points)
    # adding translation of the plane to the matrix
    # pose_rot[:,3] = - np.average(plane_points, axis=0)
    

    images, poses, bds, render_poses, i_test, scale = load_llff_data("/home/mifs/ml867/test_nerf/nerf-pytorch/data/nerf_llff_data/tea", 8,
                                                                  recenter=recenter, bd_factor=0.75,
                                                                  spherify=spherify, pose_rot=pose_rot)
    images, poses2, bds, render_poses2, i_test, scale2 = load_llff_data("/home/mifs/ml867/test_nerf/nerf-pytorch/data/nerf_llff_data/tea", 8,
                                                                  recenter=False, bd_factor=0.75,
                                                                  spherify=False, pose_rot=pose_rot)
    
    pose_rot = poses[-1]
    points = points * scale
    print(scale2)
    poses = poses[:-1] 
    poses2 = poses2[:-1]

    

    if basic:
        viz = visdom.Visdom()
        bottom_mat = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
        poses_matrix = np.concatenate([poses[:,:3,:4], bottom_mat], -2)
        camera_points = poses_matrix @ np.array([0,0,0,1])
        camera_points_direction = poses_matrix @ np.array([0,0,-0.05,1])
        best_eq, plane_points = find_plane(points)
        data = np.vstack((points , plane_points, camera_points[:, :3], camera_points_direction[:, :3]))

        data_color = [0,255,0] * np.ones_like(points)
        plane_color = [255,0,0] * np.ones_like(plane_points)
        camera_color = [0,0,255] * np.ones_like(camera_points[:, :3])
        camera_direction_color =  [255,255,255] * np.ones_like(camera_points_direction[:, :3])
        color = np.vstack((data_color, plane_color, camera_color, camera_direction_color))

        viz.scatter(data, opts=dict(markersize=2, markercolor=color, title="basic_data"))


    if plot:
        viz = visdom.Visdom()

        plane_points = points

        if not (recenter or spherify):
            pose_rot = np.concatenate([pose_rot[:3,:4], bottom], -2)
            #print(pose_rot.shape, data.shape)
            data =  np.concatenate((data, np.ones((data.shape[0],1))), axis=1)
            data = (pose_rot @ data.T).T[:,:3]

        if recenter or spherify:
            pose_rot = np.concatenate([pose_rot[:3,:4], bottom], -2)
            #print(pose_rot)
            #print(pose_rot.shape, points.shape)
            points =  np.concatenate((points, np.ones((points.shape[0],1))), axis=1)
            points = (pose_rot @ points.T).T [:,:3]
            best_eq, plane_points = find_plane(points[:,:3])
            temp = [i for i in points if i not in plane_points]
            print(np.average(temp, axis=0))
            print(np.average(plane_points, axis=0))
            # new_rot =  rotation_from_vectors(np.asarray(best_eq[0:3]), [0,0,1])
            # print(rot)
            # pose_rot = np.pad(new_rot, ((0,0), (0,1)), mode='constant')
            # pose_rot = np.concatenate([pose_rot[:3,:4], bottom], -2)
            # print(pose_rot)
            # # poses = poses @ pose_rot
            # points = (pose_rot @ points.T).T[:,:3]

        if rotate:
            # Finding camera locations and extra point in view direction
            bottom_mat = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
            poses_matrix = np.concatenate([poses[:,:3,:4], bottom_mat], -2)
            camera_points = poses_matrix @ np.array([0,0,0,1])
            camera_points_direction = poses_matrix @ np.array([0,0,-0.05,1])

            bottom_mat = np.tile(np.reshape(bottom, [1,1,4]), [render_poses.shape[0],1,1])
            poses_matrix2 = np.concatenate([render_poses[:,:3,:4], bottom_mat], -2)
            camera_points2 = poses_matrix2 @ np.array([0,0,0,1])

        if False:
            points = points @ new_rot
        camera2=np.loadtxt("output.txt")
        print(camera2.shape)
        data = np.vstack((points , plane_points, camera_points[:, :3], camera_points_direction[:, :3], camera2, camera_points2[:, :3]))
        # data = np.vstack((camera_points[:, :3], camera_points_direction[:, :3], camera2, camera_points2[:, :3]))
        data_color = [0,255,0] * np.ones_like(points)
        plane_color = [255,0,0] * np.ones_like(plane_points)
        camera_color = [0,0,255] * np.ones_like(camera_points[:, :3])
        camera_direction_color =  [255,255,255] * np.ones_like(camera_points_direction[:, :3])
        camera_color2 = [120,0,255] * np.ones_like(camera2)
        camera_direction_color2 =  [255,255,255] * np.ones_like(camera_points2[:, :3])

        color = np.vstack((data_color, plane_color, camera_color, camera_direction_color, camera_color2, camera_direction_color2))
        # color = np.vstack((camera_color, camera_direction_color, camera_color2, camera_direction_color2))

        viz.scatter(data, opts=dict(markersize=1, markercolor=color, title="Rotation + recenter. Spherify = {}, Ceneter = {}".format(spherify, recenter)))

"""This seems to get correct camera positions! (for no spherification and recentering)"""




if __name__=='__main__':
    main()