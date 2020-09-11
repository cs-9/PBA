import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from time import time
import json
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import trimesh
######################################### FUNCTIONS ###########################################################

SAMPLE_SIZE = 10

def valid(l):
    for x in l:
        if x != 0:
            return True
    return False

def return_camera_info(file1,file2):
    '''
    Reads json file and returns camera parameters and locations
    '''
    f1 = open(file1, "r")
    data_RC = json.load(f1)['params']

    f2 = open(file2, "r")
    data_P = json.load(f2)['data']

    data_P = [x for x in data_P if x['height'] != 0 and valid(x['P']) ]
    files = [x['name'] for x in data_P]

    # Camera details
    P = np.array([np.array(x['P']).reshape([3, 4]) for x in data_P])
    R = np.array([np.array(x['R']).reshape([3, 3]) for x in data_RC])
    C = np.array([np.array(x['C']) for x in data_RC])
    K = np.matmul(P[:, :, :3], np.linalg.inv(R))

    return (P,R,C,K,files)

def fetch_mesh_coordinates(mesh_coords):
    '''
    Fetches coordinates,normal,area of the meshes
    '''
    p1 = np.array(list(mesh.elements[0].data[mesh_coords[0][0]]))
    p2 = np.array(list(mesh.elements[0].data[mesh_coords[0][1]]))
    p3 = np.array(list(mesh.elements[0].data[mesh_coords[0][2]]))
    a1 = p2-p1
    a2 = p3-p1
    cross_product = np.cross(a1,a2)
    norm_cross_product = np.linalg.norm(cross_product)
    #Normal vector
    n =  cross_product/norm_cross_product
    #Area
    area = (1/2)*norm_cross_product
    return ((p1,p2,p3),n,area)

def fetch_area_normal(pts):
    '''
    Fetches area and normal of the triangle formed by the points
    '''
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    a1 = p2-p1
    a2 = p3-p1
    cross_product = np.cross(a1,a2)
    norm_cross_product = np.linalg.norm(cross_product)
    #Normal vector
    n =  cross_product/norm_cross_product
    #Area
    area = (1/2)*norm_cross_product
    return (n,area)

def SignedVolume(a,b,c,d):
    return (1.0/6.0)*np.dot(np.cross(b-a,c-a),d-a)

def intersecting_mesh(p1,p2,p3,q1,q2):
    """Finds if vector formed by q1,q2 intersects the triangle made by p1,p2,p3. Returns 1 if there is intersection else 0"""
    A1 = np.sign(SignedVolume(q1,p1,p2,p3))
    A2 = np.sign(SignedVolume(q2,p1,p2,p3))
    B1 = np.sign(SignedVolume(q1,q2,p1,p2))
    B2 = np.sign(SignedVolume(q1,q2,p2,p3))
    B3 = np.sign(SignedVolume(q1,q2,p3,p1))
    if (A1!=A2) and (B1==B2==B3):
        return 1
    else:
        return 0
    
def visibility(mesh_index,camera_loc):
    '''Calculates visibilty for a single mesh based on intersecting meshes'''
    (p1,p2,p3),n,area=Mesh_info[mesh_index]
    q1 = camera_loc
    q2 = (p1+p2+p3)/3
    vector = vector = (q2-q1)/np.linalg.norm(q2-q1)
    #Calculates points on the vector joining
    Mesh_vector = q1+ np.array([[list(np.linalg.norm((mesh_centroid-q1),axis=1)),]*3]).T[:,:,0]*vector
    dist = np.linalg.norm(np.mean(Mesh_vertices[mesh_index,:],axis=0)-Mesh_vertices[mesh_index,0])    
    Search_mesh = np.where((np.abs(Mesh_vector[:,0]-mesh_centroid[:,0])<=dist) & (np.abs(Mesh_vector[:,1]-mesh_centroid[:,1])<=dist) & (np.abs(Mesh_vector[:,2]-mesh_centroid[:,2])<=dist))[0] 
    #Modifying q2 so that it is just outside the triangle
    q2 = q2 + 0.00001*(q2-q1)/np.linalg.norm(q2-q1)
    #Visibility function
    visibility = []
    main_mesh = mesh_index
    for mesh_index in Search_mesh:
        (p1,p2,p3),n,area = Mesh_info[mesh_index]
        visibility.append(intersecting_mesh(p1,p2,p3,q1,q2))
    a = Search_mesh[np.where(np.array(visibility)==1)[0]]
    if len(a)>=2:
        return 0
    else: 
        return 1

def calculate_visibility_mesh(camera_locations):
    '''Calculates visibility for all cameras and meshes, indexed as Camera_visibility[camera][mesh]'''
    start = time()
    Camera_visibility = []
    for camera in tqdm(range(len(camera_locations)),desc="Calculating visibility for given list of {} cameras".format(len(camera_locations))):
        Mesh_visibility = np.zeros(len(mesh_centroid))
        Mesh_list = np.arange(len(mesh_centroid))
        camera_loc = np.array(list(camera_locations[camera]))
        Mesh_vertices_calc = Mesh_vertices.copy()
        while len(Mesh_list)>=1:
    #         print("{} meshes left....".format(len(Mesh_list)))
            mesh_index = Mesh_list[0]
            a = visibility(mesh_index,camera_loc)
            #Find neighbouring meshes
            indices = np.unique(np.where((Mesh_vertices_calc==Mesh_vertices[mesh_index][0]) | (Mesh_vertices_calc==Mesh_vertices[mesh_index][1]) | (Mesh_vertices_calc==Mesh_vertices[mesh_index][2]))[0])
            #Update the visibility as the one of the selected mesh   
            Mesh_visibility[Mesh_list[indices]] = a
            #Update the mesh list
            Mesh_list = np.delete(Mesh_list, indices)
            #Update the vertices list
            Mesh_vertices_calc = np.delete(Mesh_vertices_calc, indices,axis=0)
        Camera_visibility.append(Mesh_visibility)
    end = time()
    print("Done in {}sec".format(end-start))
    return Camera_visibility

def Energy_function_calc(Vertex_gradient,texture_linear_0,sample_size=SAMPLE_SIZE):
    '''Finds Energy value for the whole mesh and camera'''
    #Calculates barycentric coordinates
    points = []
    for u in np.arange(0,1,1/sample_size):
        for v in np.arange(0,1-u,1/sample_size):
            w = 1-u-v
            points.append([u,v,w])
    points = np.array(points)
    h,w,_ = np.shape(Images[0])
    integration = 0
    Energy_over_mesh = np.zeros(len(Mesh_info))
    
    for camera_index in tqdm(range(len(camera_locations)),desc="Calculating photometric loss...",position=0, leave=True):
    # for camera_index in range(len(camera_locations)):
        f_x = K[camera_index][0,0]
        f_y = K[camera_index][1,1]
        visibility = Camera_visibility[camera_index]
        P_camera = P[camera_index]
        C = np.array(list(camera_locations[camera_index]))
        for mesh_id in range(len(Mesh_info)):
            if visibility[mesh_id]!=0:    
                X_j = Vertex_gradient[mesh.elements[1].data[mesh_id][0]] 
                n_j,A_j = fetch_area_normal(X_j)
                #P=uA+vB+wC
                X_u = np.dot(points,X_j)
                #Finds x = PX
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                #Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                #For alpha term
                d = X_u - C
                d_z = np.linalg.norm(d,axis=1)
                alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                #Final integeration over a single meshe
                temp_val = np.abs(A_j*np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]
                Energy_over_mesh[mesh_id] = Energy_over_mesh[mesh_id] + temp_val
                integration += temp_val
    return integration,Energy_over_mesh

def Numerical_gradient_mesh(vertex_id,Vertex_grad,texture_linear_0,diff=0.0000001,sample_size=SAMPLE_SIZE):
    '''
    Calculates numerical gradient for a single vertex
    Central diffference Method used: (f(x+diff)-f(x-diff))/2*diff
    Sample_size: Number of sampling barycentric coordinates in a triangle N = sample_size*(sample_size+1)/2
    Since only the meshes surronding the point gets affected, hence only the subtraction of energies
    of those points taken
    '''
    #Calculates barycentric coordinates
    points = []
    for u in np.arange(0,1,1/sample_size):
        for v in np.arange(0,1-u,1/sample_size):
            w = 1-u-v
            points.append([u,v,w])
    points = np.array(points)
    h,w,_ = np.shape(Images[0])
    affected_meshes = np.unique(np.where((Mesh_vertices==Vertex[vertex_id]))[0])
    gradient = np.array([0.0,0.0,0.0]) 
    for axis in range(3):
        Vertex_gradient = Vertex_grad.copy()
        integration = 0
        for camera_index in range(len(camera_locations)):
            f_x = K[camera_index][0,0]
            f_y = K[camera_index][1,1]
            visibility = Camera_visibility[camera_index]
            P_camera = P[camera_index]
            C = np.array(list(camera_locations[camera_index]))
            for mesh_id in affected_meshes:
                #Positive perturbation f(x+diff)
                if visibility[mesh_id]!=0:    
                    Vertex_gradient[vertex_id][axis] = Vertex_grad[vertex_id][axis]+diff
                    X_j = Vertex_gradient[mesh.elements[1].data[mesh_id][0]] 
                    n_j,A_j = fetch_area_normal(X_j)
                    #P=uA+vB+wC
                    X_u = np.dot(points,X_j)
                    #Finds x = PX
                    se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                    se = np.divide(se,se[:,-1][:,np.newaxis])
                    #Texture term shd be added, term from image fetched
                    image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                    #For alpha term
                    d = X_u - C
                    d_z = np.linalg.norm(d,axis=1)
                    alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                    #Final integeration over a single meshe
                    integration_pos = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]

                    #Negative perturbation f(x-diff)
                    Vertex_gradient[vertex_id][axis] = Vertex[vertex_id][axis]-diff
                    X_j = Vertex_gradient[mesh.elements[1].data[mesh_id][0]] 
                    n_j,A_j = fetch_area_normal(X_j)
                    #P=uA+vB+wC
                    X_u = np.dot(points,X_j)
                    #Finds x = PX
                    se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                    se = np.divide(se,se[:,-1][:,np.newaxis])
                    #Texture term shd be added, term from image fetched
                    image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                    #For alpha term
                    d = X_u - C
                    d_z = np.linalg.norm(d,axis=1)
                    alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                    #Final integeration over a single meshe
                    integration_neg = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]

                    #Final addition 
                    integration += (integration_pos-integration_neg)
        gradient[axis] = integration/(2*diff)
    return gradient

def write_to_PLY(Vertex_update,Energy_function, fName="mesh_visualize.ply"):
    '''
    Writes the vertex and faces into PLY file with the help of Vertex_update. The meshes will
    have color according to the defined energy function over the meshes. Can be texture too
    Parameters:
        Vertex_update: Array of vertexes
        Energy_function: Value of energy over different meshes
    Returns:
        ply file named mesh_visualize.ply
    '''
    #Defining colormap
    cmap = plt.cm.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=min(Energy_function), vmax=max(Energy_function))
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    coloring = (255*scalarMap.to_rgba(Energy_function)).astype(int)
    #Vertices
    Normal_vertices = [(Vertex_update[i][0],Vertex_update[i][1],Vertex_update[i][2]) for i in range(len(Vertex_update))]
    camera_vertices = [(camera_locations[i][0],camera_locations[i][1],camera_locations[i][2]) for i in range(len(camera_locations))]
    Write_vertices = Normal_vertices + camera_vertices
    #Faces
    coloring = (255*scalarMap.to_rgba(Energy_function)).astype(int)
    Colored_Mesh = [(3,mesh.elements[1].data[i][0][0],mesh.elements[1].data[i][0][1],mesh.elements[1].data[i][0][2],coloring[i][0],coloring[i][1],coloring[i][2]) for i in range(len(mesh.elements[1].data))]
    num_vertices = len(Write_vertices)
    #Writing to PLY file
    num_vertices = len(Write_vertices)
    num_faces = len(Colored_Mesh)
    header_lines = "ply\nformat ascii 1.0\ncomment meshes colored according to function\nelement vertex {}\ncomment modified vertices\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".format(num_vertices,num_faces)
    for i in range(num_vertices):
        header_lines = header_lines + str(Write_vertices[i]).replace(",","")[1:-1] + '\n'
    for i in range(num_faces):
        header_lines = header_lines + str(Colored_Mesh[i]).replace(",","")[1:-1] + '\n'
    with open(fName,"w") as f:
        f.write(header_lines)
        

def init_texture():
    A, B, C = Mesh_vertices[:, 0], Mesh_vertices[:, 1], Mesh_vertices[:, 2]
    A = np.hstack([A, np.ones([A.shape[0], 1], dtype=int)])
    B = np.hstack([B, np.ones([B.shape[0], 1], dtype=int)])
    C = np.hstack([C, np.ones([C.shape[0], 1], dtype=int)])

    texture = np.zeros([mesh.elements[1].data.shape[0], SAMPLE_SIZE * SAMPLE_SIZE, 3], dtype=np.float64)
    wt = np.zeros([mesh.elements[1].data.shape[0], SAMPLE_SIZE * SAMPLE_SIZE], dtype=float)
    print("Computing texture")
    for cam in tqdm(range(len(image_files)),desc="Calculating texture {} ".format(len(image_files))):
        p = P[cam]
        img = images_0[cam]
        # alpha = K[cam][0][0] * K[cam][1][1]
        for i in range(mesh.elements[1].data.shape[0]):
            if not visibility_0[cam][i]:
                continue
            a, b, c = A[i], B[i], C[i]
            a_, b_, c_ = p.dot(a.T), p.dot(b.T), p.dot(c.T)
            a_, b_, c_ = a_ / a_[-1], b_/b_[-1], c_/c_[-1]
            for u in np.arange(0, 1, 1/SAMPLE_SIZE):
                for v in np.arange(0, 1 - u, 1/SAMPLE_SIZE):
                    # u, v, w
                    w = 1 - u - v
                    pos = np.round(u * a_ + v * b_ + w * c_)
                    if pos[0] >= img.shape[1] or pos[1] >= img.shape[0] or pos[0] < 0 or pos[1] < 0:
                        continue
                    # d = (u * a + v * b + w * c)[:3] - CamCenter[cam]
                    # n_j, A_j = fetch_area_normal([a[:3], b[:3], c[:3]])
                    # wt_temp = np.abs(n_j.dot(d) * alpha * (10 ** -9) / (np.linalg.norm(d) ** 3))
                    wt_temp = 1
                    # if wt_temp == 0 or np.isnan(wt_temp):
                    #     continue
                    x_coord, y_coord = pos[1], pos[0]
                    wt[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)] += wt_temp
                    texture[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)] += img[int(x_coord)][int(y_coord)] * wt_temp
    return texture, wt


def show_texture(texture, wt, fileName):
    points = []
    colors = []

    for i in range(mesh.elements[1].data.shape[0]):
        A, B, C = Vertex[tuple(mesh.elements[1].data[i])]
        for u in np.arange(0, 1, 1/SAMPLE_SIZE):
            for v in np.arange(0, 1 - u, 1/SAMPLE_SIZE):
                if wt[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)] == 0:
                    continue
                w = 1 - u - v
                X = u * A + v * B + w * C
                points.append(X)
                color = texture[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)] / wt[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)]
                colors.append([int(color[2]), int(color[1]), int(color[0])])
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.export(fileName)
    

def linearize_texture(texture):
    texture_mesh = []
    for i in range(mesh.elements[1].data.shape[0]):
        texture_linear = []
        for u in np.arange(0, 1, 1/SAMPLE_SIZE):
            for v in np.arange(0, 1 - u, 1/SAMPLE_SIZE):
                texture_linear.append(texture[i][int(u * SAMPLE_SIZE * SAMPLE_SIZE + v * SAMPLE_SIZE)])
        texture_mesh.append(texture_linear)
    return texture_mesh

def gradient_descent(Epochs,Vertices_grad,texture_linear_0,learning_rate,verbosity=1):
    print("Starting gradient descent....")
    global K, R, camera_locations, P, texture_0, wt_0
    loss,_ = Energy_function_calc(Vertices_grad,texture_linear_0)
    for epoch in range(Epochs):
        print("Epoch:{}-----> Photometric Loss:{}".format(epoch,loss))
        for vertex_id in range(len(Vertices_grad)):
            Vertices_grad[vertex_id] = Vertices_grad[vertex_id] - learning_rate*Numerical_gradient_mesh(vertex_id,Vertices_grad,texture_linear_0)
        for cameraId in range(len(image_files)):
            grad_K, grad_R, grad_C = Numerical_gradient_cam(cameraId, Vertices_grad, texture_linear_0)
            K[cameraId] = K[cameraId] - learning_rate * grad_K
            grad_R = learning_rate * grad_R
            TotalRotation = np.array([
                [
                    [1, 0, 0],
                    [0, np.cos(grad_R[0]), -np.sin(grad_R[0])],
                    [0, np.sin(grad_R[0]), np.cos(grad_R[0])]
                ],
                [
                    [np.cos(grad_R[1]), 0, np.sin(grad_R[1])],
                    [0, 1, 0],
                    [-np.sin(grad_R[1]), 0, np.cos(grad_R[1])]
                ],
                [
                    [np.cos(grad_R[2]), -np.sin(grad_R[2]), 0],
                    [np.sin(grad_R[2]), np.cos(grad_R[2]), 0],
                    [0, 0, 1]
                ]
            ])
            R[cameraId] = R[cameraId].dot(TotalRotation[0]).dot(TotalRotation[1]).dot(TotalRotation[2])
            camera_locations[cameraId] = camera_locations[cameraId] - learning_rate * grad_C
            print(grad_C)
            P[cameraId][:, :3] = K[cameraId].dot(R[cameraId])
            P[cameraId][:, 3] = - K[cameraId].dot(R[cameraId]).dot(camera_locations[cameraId])
        # texture_0, wt_0 = update_texture(texture_0, wt_0)
    if epoch%verbosity==0:
        loss,_= Energy_function_calc(Vertices_grad,texture_linear_0)
        # show_texture(texture_0, wt_0, epoch)
    np.save("camera_K_params", K)
    np.save("camera_rot_params", R)
    np.save("camera_center_params", camera_locations)
    return Vertices_grad


def Numerical_gradient_cam(camera_id, vertices, texture_linear_0, diff=1e-7, sample_size=SAMPLE_SIZE):
    '''
    Calculates numerical gradient for a single camera
    Central diffference Method used: (f(x+diff)-f(x-diff))/2*diff
    Sample_size: Number of sampling barycentric coordinates in a triangle N = sample_size*(sample_size+1)/2
    '''
    # Calculates barycentric coordinates
    points = []
    for u in np.arange(0,1,1/sample_size):
        for v in np.arange(0,1-u,1/sample_size):
            w = 1-u-v
            points.append([u,v,w])
    points = np.array(points)
    h,w,_ = np.shape(Images[0])
    gradient_K = np.zeros([3, 3], dtype=float)
    gradient_R = np.zeros([3], dtype=float)
    gradient_C = np.array([0., 0., 0.])
    # compute gradient K
    k_index_map = {0 : (0, 0), 1 : (0, 1), 2 : (0, 2), 3 : (1, 1), 4 : (1, 2)}
    
    f_x = K[camera_id][0][0]
    f_y = K[camera_id][1][1]
    visibility = Camera_visibility[camera_id]
    K_camera = K[camera_id]
    R_camera = R[camera_id]
    C_camera = camera_locations[camera_id]
    # Compute gradient K
    for axis in range(5):
        integration = 0
        K_diff = np.zeros([3, 3], dtype=float)
        K_diff[k_index_map[axis]] += diff
        P_camera_p = np.zeros([3, 4], dtype=float)
        P_camera_n = np.zeros([3, 4], dtype=float)
        P_camera_p[:, :3] = (K_camera + K_diff).dot(R_camera)
        P_camera_p[:, 3] = -(K_camera + K_diff).dot(R_camera).dot(C_camera)
        P_camera_n[:, :3] = (K_camera - K_diff).dot(R_camera)
        P_camera_n[:, 3] = -(K_camera - K_diff).dot(R_camera).dot(C_camera)
        for mesh_id in range(mesh.elements[1].data.shape[0]):
            if visibility[mesh_id] != 0:
                X_j = vertices[tuple(mesh.elements[1].data[mesh_id])]
                n_j, A_j = fetch_area_normal(X_j)
                # X_u = uA + vB + wC
                X_u = np.dot(points, X_j)
                # For alpha term
                d = X_u - C_camera
                d_z = np.linalg.norm(d, axis=1)
                alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                # positibe perturbation
                # Find x = PX
                se = np.dot(P_camera_p, np.hstack([X_u, np.ones([len(X_u), 1])]).T).T
                se = np.divide(se, se[:, -1][:, np.newaxis])
                #Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                # Integration
                integration_pos = A_j * np.abs(np.dot(image_term, np.dot(alpha, n_j))) * visibility[mesh_id]
                # negative perturbation
                # Find x = PX
                se = np.dot(P_camera_n, np.hstack([X_u, np.ones([len(X_u), 1])]).T).T
                se = np.divide(se, se[:, -1][:, np.newaxis])
                #Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                # Integration
                integration_neg = A_j * np.abs(np.dot(image_term, np.dot(alpha, n_j))) * visibility[mesh_id]
                integration += integration_pos - integration_neg
        gradient_K[k_index_map[axis]] = integration / (2 * diff)
    
    # Compute gradient R
    # R = Rx() * Ry() * Rz()
    diff = 1e-5
    R_diffs = np.array([
        [
            [1, 0, 0],
            [0, np.cos(diff), -np.sin(diff)],
            [0, np.sin(diff), np.cos(diff)]
        ],
        [
            [np.cos(diff), 0, np.sin(diff)],
            [0, 1, 0],
            [-np.sin(diff), 0, np.cos(diff)]
        ],
        [
            [np.cos(diff), -np.sin(diff), 0],
            [np.sin(diff), np.cos(diff), 0],
            [0, 0, 1]
        ]
    ])
    P_camera = np.zeros([3, 4], dtype=float)
    for axis in range(3):
        for mesh_id in range(mesh.elements[1].data.shape[0]):
            if visibility[mesh_id] != 0:
                X_j = vertices[tuple(mesh.elements[1].data[mesh_id])]
                n_j, A_j = fetch_area_normal(X_j)
                # X_u = uA + vB + wC
                X_u = np.dot(points, X_j)
                # For alpha term
                d = X_u - C_camera
                d_z = np.linalg.norm(d, axis=1)
                alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                # positibe perturbation
                P_camera[:, :3] = K_camera.dot(R_camera.dot(R_diffs[axis]))
                P_camera[:, 3] = -K_camera.dot(R_camera.dot(R_diffs[axis])).dot(C_camera)
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                # Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                #Final integeration over a single meshe
                integration_pos = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]

                # negative perturbation
                P_camera[:, :3] = K_camera.dot(R_camera.dot(-R_diffs[axis]))
                P_camera[:, 3] = -K_camera.dot(R_camera.dot(-R_diffs[axis])).dot(C_camera)
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                # Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                #Final integeration over a single meshe
                integration_neg = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]
                integration += integration_pos - integration_neg
        gradient_R[axis] = integration / (2 * diff)
    
    # Compute gradient C
    diff = 1e-8
    P_camera = np.zeros([3, 4], dtype=float)
    P_camera[:, :3] = K_camera.dot(R_camera)
    for axis in range(3):
        C_diff = np.zeros([3], dtype=float)        
        C_diff[axis] = diff
        integration = 0
        for mesh_id in range(mesh.elements[1].data.shape[0]):
            # TODO: SHOULD USE UPDATED VISIBILITY HERE
            if visibility[mesh_id] != 0:
                X_j = vertices[tuple(mesh.elements[1].data[mesh_id])]
                n_j, A_j = fetch_area_normal(X_j)
                # X_u = uA + vB + wC
                X_u = np.dot(points, X_j)
                # For alpha term
                d = X_u - C_camera
                d_z = np.linalg.norm(d, axis=1)
                alpha = (10**-10)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)

                # positibe perturbation
                P_camera[:, 3] = -K_camera.dot(R_camera).dot(C_camera + C_diff)
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                # Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                #Final integeration over a single meshe
                integration_pos = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]

                # negative perturbation
                P_camera[:, 3] = -K_camera.dot(R_camera).dot(C_camera - C_diff)
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                # Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_id][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)]-np.array(texture_linear_0[mesh_id]),axis=1)
                #Final integeration over a single meshe
                integration_neg = A_j*np.abs(np.dot(image_term,np.dot(alpha,n_j)))*visibility[mesh_id]
                integration += integration_pos - integration_neg
        gradient_C[axis] = integration / (2 * diff)
    
#     print(gradient_K, gradient_R, gradient_C)
    return gradient_K, gradient_R, gradient_C
    

#########################################################################################################################
#Fetch initial scene data
P,R,camera_locations,K,files = return_camera_info("./SFM_results_dataset/camera_params.test","./SFM_results_dataset/images.test")
mesh = PlyData.read('./SFM_results_dataset/scene_dense_mesh_refine.ply')
#Load all images for each camera
operating_dir = "./datasets/templeRing/"
Images = [cv2.imread(operating_dir+files[i].split("/")[-1]) for i in range(len(camera_locations))]

#Extract mesh related information
Mesh_info = []
for i in tqdm(range(len(mesh.elements[1].data)),desc="Fetching mesh related information..."):
    Mesh_info.append(fetch_mesh_coordinates(mesh.elements[1].data[i]))
#Coordinates of vertex each mesh
Mesh_vertices = np.array([[i[0][0],i[0][1],i[0][2]]for i in Mesh_info])
#Centroid of mesh
mesh_centroid = np.mean(Mesh_vertices,axis=1)
#Array of vertices
Vertex = np.array([np.array(list(mesh.elements[0].data[i])) for i in range(len(mesh.elements[0].data))])


###########################################
### TEXTURE STUFF
start_0 = time()
CamCenter = camera_locations
image_files = [operating_dir+files[i].split("/")[-1] for i in range(len(camera_locations))]
images_0 = Images
#Visibility table
Camera_visibility = calculate_visibility_mesh(camera_locations)
visibility_0 =  Camera_visibility

texture_0, wt_0 = init_texture()
texture_linear_0 = linearize_texture(texture_0)
show_texture(texture_0, wt_0, "textured_mesh_before.ply")
###########################################
integeration,Energy_over_mesh = Energy_function_calc(Vertex,texture_linear_0)
print("Total photometric loss :{}".format(integeration))
write_to_PLY(Vertex,Energy_over_mesh, "mesh_visualize_before.ply")
grad = Numerical_gradient_mesh(34,Vertex,texture_linear_0)
print("Gradient:{}".format(grad))

#Gradient descent over mesh coordinates
np.save("vertex_before", Vertex)
Vertex = gradient_descent(15,Vertex,texture_linear_0, 0.0001)
np.save("vertex_after", Vertex)
integeration,Energy_over_mesh = Energy_function_calc(Vertex,texture_linear_0)
print("Total photometric loss :{}".format(integeration))

print("Saving into PLY file....")
show_texture(texture_0, wt_0, "textured_mesh_after.ply")
write_to_PLY(Vertex,Energy_over_mesh, "mesh_visualize_after.ply")

