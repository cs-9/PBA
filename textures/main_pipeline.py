import numpy as np
from time import time
import json
import trimesh
import matplotlib.image as mplimg

######################################### FUNCTIONS ###################################################
def valid(l):
    for x in l:
        if x != 0:
            return True
    return False


def return_camera_info(file1, file2):
    '''
    Reads json file and returns camera parameters and locations
    '''
    f1 = open(file1, "r")
    data_RC = json.load(f1)['params']

    f2 = open(file2, "r")
    data_P = json.load(f2)['data']

    data_P = [x for x in data_P if x['height'] != 0 and valid(x['P'])]
    files = [x['name'] for x in data_P]

    # Camera details
    P = np.array([np.array(x['P']).reshape([3, 4]) for x in data_P])
    R = np.array([np.array(x['R']).reshape([3, 3]) for x in data_RC])
    C = np.array([np.array(x['C']) for x in data_RC])
    K = np.matmul(P[:, :, :3], np.linalg.inv(R))

    return P, R, C, K, files


##########################################################################################################
# Fetch initial scene data
start = time()
P, R, CamCenter, K, image_files = return_camera_info("camera_params.test", "images.test")
operating_dir = "../datasets/templeRing/"
image_files = [operating_dir+image_files[i].split("/")[-1] for i in range(len(CamCenter))]
mesh = trimesh.load('scene_dense_mesh.ply')

centroids = mesh.vertices[mesh.faces].mean(axis=1)
vertices = mesh.vertices[mesh.faces]
normals = np.cross(vertices[:, 2] - vertices[:, 0], vertices[:, 1] - vertices[:, 0])

# Saved visibility to a file to save computation time
visibility = np.load("visibility_cache.tmp.npy")

width = 640
height = 480
reso = 3
def init_texture():
    A, B, C = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    A = np.hstack([A, np.ones([A.shape[0], 1], dtype=int)])
    B = np.hstack([B, np.ones([B.shape[0], 1], dtype=int)])
    C = np.hstack([C, np.ones([C.shape[0], 1], dtype=int)])

    start = time()

    texture = np.zeros([mesh.faces.shape[0], reso * reso, 3], dtype=np.float64)
    wt = np.zeros([mesh.faces.shape[0]], dtype=float)

    for cam in range(len(image_files)):
        p = P[cam]
        img = mplimg.imread(image_files[cam])
        alpha = K[cam][0][0] * K[cam][1][1]
        for i in range(vertices.shape[0]):
            if not visibility[cam][i]:
                continue
            a, b, c = A[i], B[i], C[i]
            a_, b_, c_ = p.dot(a.T), p.dot(b.T), p.dot(c.T)
            a_, b_, c_ = a_ / a_[-1], b_/b_[-1], c_/c_[-1]
            b = b - a
            c = c - a
            for u in range(reso):
                for v in range(0, reso - u):
                    pos = np.round(a_ + u / reso * b_ + v / reso * c_)
                    if pos[0] >= 480 or pos[1] >= 640:
                        continue
                    d = (a + u * b + v * c)[:3] - CamCenter[cam]
                    wt_temp = np.abs(normals[i].dot(d) * alpha / (d[-1] ** 3))
                    if wt_temp == 0 or np.isnan(wt_temp):
                        continue
                    wt[i] += wt_temp
                    x_coord, y_coord = pos[0], pos[1]
                    texture[i][u * reso + v] += img[int(x_coord)][int(y_coord)] * wt_temp
    return texture, wt

def show_texture(texture, wt):
    points = []
    colors = []

    for i in range(mesh.faces.shape[0]):
        if wt[i] == 0:
            continue
        temp = texture[i] / wt[i]
        A, B, C = vertices[i]
        for u in range(reso):
            for v in range(reso - u):
                X = A + u * (B - A) + v * (C - A)
                points.append(X)
                colors.append(temp[u * reso + v])
    pcd = trimesh.PointCloud(vertices=points, colors=colors)

    print("Time taken:", time() - start)
    pcd.scene().show()

def update_texture(NUM_ITERS):
    # TEXTURE UPDATES - For some reason generated colors are BAD
    # dt = 1e-7
    # wt = np.zeros([mesh.faces.shape[0]], dtype=float)
    #
    # for _ in range(NUM_ITERS):
    #     texture_2 = texture.copy()
    #     for cam in range(len(image_files)):
    #         print("%.2f%% Complete. Time Taken: %.2fs" % (100 * (cam / len(image_files) + _) / NUM_ITERS, time() - start))
    #         p = P[cam]
    #         img = mplimg.imread(image_files[cam])
    #         alpha = K[cam][0][0] * K[cam][1][1]
    #         for i in range(vertices.shape[0]):
    #             if not visibility[cam][i]:
    #                 continue
    #             a, b, c = A[i], B[i], C[i]
    #             a_, b_, c_ = p.dot(a.T), p.dot(b.T), p.dot(c.T)
    #             a_, b_, c_ = a_ / a_[-1], b_/b_[-1], c_/c_[-1]
    #             b = b - a
    #             c = c - a
    #             for u in range(reso):
    #                 for v in range(0, reso - u):
    #                     pos = np.round(a_ + u / reso * b_ + v / reso * c_)
    #                     if pos[0] >= 480 or pos[1] >= 640:
    #                         continue
    #                     d = (a + u * b + v * c)[:3] - CamCenter[cam]
    #                     wt_temp = np.abs(normals[i].dot(d) * alpha / (d[-1] ** 3))
    #                     if wt_temp == 0 or np.isnan(wt_temp):
    #                         continue
    #                     x_coord, y_coord = pos[0], pos[1]
    #                     texture_2[i][u * reso + v] += dt * (img[int(x_coord)][int(y_coord)] - texture[i][u * reso + v]) * wt_temp
    #                     wt[i] += wt_temp
    #     texture = texture_2
    return
t, wts = init_texture()
update_texture(5)
show_texture(t, wts)
