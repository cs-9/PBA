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

def Energy_function_calc(sample_size=14):
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
    
    for camera_index in tqdm(range(len(camera_locations)),desc="Calculating energy function..."):
        f_x = K[camera_index][0,0]
        f_y = K[camera_index][1,1]
        visibility = Camera_visibility[camera_index]
        P_camera = P[camera_index]
        C = np.array(list(camera_locations[camera_index]))
        for mesh_id in range(len(Mesh_info)):
            if visibility[mesh_id]!=0:    
                X_j = Mesh_vertices[mesh_id] 
                _,n_j,A_j = Mesh_info[mesh_id]
                #P=uA+vB+wC
                X_u = np.dot(points,X_j)
                #Finds x = PX
                se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                se = np.divide(se,se[:,-1][:,np.newaxis])
                #Texture term shd be added, term from image fetched
                image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)],axis=1)
                #For alpha term
                d = X_u - C
                d_z = np.linalg.norm(d,axis=1)
                alpha = (10**-6)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                #Final integeration over a single meshe
                temp_val = A_j*np.dot(image_term,np.dot(alpha,n_j))*visibility[mesh_id]
                Energy_over_mesh[mesh_id] = Energy_over_mesh[mesh_id] + temp_val
                integration += temp_val
    return integration,Energy_over_mesh

def Numerical_gradient_mesh(vertex_id,diff=0.0000001,sample_size=14):
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
        Vertex_gradient = Vertex.copy()
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
                    Vertex_gradient[vertex_id][axis] = Vertex[vertex_id][axis]+diff
                    X_j = Vertex_gradient[mesh.elements[1].data[mesh_id][0]] 
                    n_j,A_j = fetch_area_normal(X_j)
                    #P=uA+vB+wC
                    X_u = np.dot(points,X_j)
                    #Finds x = PX
                    se = np.dot(P_camera,np.hstack((X_u,np.ones((len(X_u),1)))).T).T
                    se = np.divide(se,se[:,-1][:,np.newaxis])
                    #Texture term shd be added, term from image fetched
                    image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)],axis=1)
                    #For alpha term
                    d = X_u - C
                    d_z = np.linalg.norm(d,axis=1)
                    alpha = (10**-7)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                    #Final integeration over a single meshe
                    integration_pos = A_j*np.dot(image_term,np.dot(alpha,n_j))*visibility[mesh_id]

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
                    image_term = np.linalg.norm(Images[camera_index][np.clip(se[:,1].astype(int),0,h-1),np.clip(se[:,0].astype(int),0,w-1)],axis=1)
                    #For alpha term
                    d = X_u - C
                    d_z = np.linalg.norm(d,axis=1)
                    alpha = (10**-7)*(f_x*f_y)*np.divide(d,d_z[:,np.newaxis]**3)
                    #Final integeration over a single meshe
                    integration_neg = A_j*np.dot(image_term,np.dot(alpha,n_j))*visibility[mesh_id]

                    #Final addition 
                    integration += (integration_pos-integration_neg)
        gradient[axis] = integration/(2*diff)
    return gradient

def write_to_PLY(Vertex_update,Energy_function):
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
    with open("./mesh_visualize.ply","w") as f:
        f.write(header_lines)
        
        
def init_texture_coords():
    A, B, C = vertices_0[:, 0], vertices_0[:, 1], vertices_0[:, 2]
    A = np.hstack([A, np.ones([A.shape[0], 1], dtype=int)])
    B = np.hstack([B, np.ones([B.shape[0], 1], dtype=int)])
    C = np.hstack([C, np.ones([C.shape[0], 1], dtype=int)])

    start = time()

    texture_coords = np.zeros([mesh_0.faces.shape[0], len(image_files), 3, 2], dtype=np.float64)
    b_img_tex = np.zeros([mesh_0.faces.shape[0], len(image_files)], dtype=bool)

    for cam in range(len(image_files)):
        # print("%.2f%% Complete. Time Taken: %.2fs" % (100 * cam / len(image_files), time() - start))
        p = P[cam]
        img = images_0[cam]
        for i in range(vertices_0.shape[0]):
            if not visibility_0[cam][i]:
                continue
            a, b, c = A[i], B[i], C[i]
            a_, b_, c_ = p.dot(a.T), p.dot(b.T), p.dot(c.T)
            a_, b_, c_ = a_ / a_[-1], b_ / b_[-1], c_ / c_[-1]
            a__ = np.round(a_)
            b__ = np.round(b_)
            c__ = np.round(c_)
            if (0 <= int(a__[1]) < img.shape[0]) \
                    and (0 <= int(b__[1]) < img.shape[0]) \
                    and (0 <= int(c__[1]) < img.shape[0]) \
                    and (0 <= int(a__[0]) < img.shape[1]) \
                    and (0 <= int(b__[0]) < img.shape[1]) \
                    and (0 <= int(c__[0]) < img.shape[1]):
                b_img_tex[i][cam] = True
                texture_coords[i][cam][0] = a__[:2]
                texture_coords[i][cam][1] = b__[:2]
                texture_coords[i][cam][2] = c__[:2]
    return texture_coords, b_img_tex


def getTex(faceId, X_, u=-1., v=-1.):
    A, B, C = vertices_0[faceId]
    if u == -1 or v == -1:
        X = X_ - A
        B -= A
        C -= A
        v = (X[1] * B[0] - B[1] * X[0]) / (B[0] * C[1] - B[1] * C[0])
        u = (X[0] - v * C[0]) / B[0]
    res_tex = np.array([0., 0., 0.])
    w = 0
    for cam in range(len(images_0)):
        if not b_img_tex_0[faceId][cam]:
            continue
        a, b, c = texture_coords_0[faceId][cam]
        b, c = b - a, c - a
        coords = a + u * b + v * c
        d = X_ - CamCenter[cam]
        # w_temp = K[cam][0][0] * K[cam][1][1] * np.dot(normals_0[faceId], d) / (np.linalg.norm(d) ** 3)
        w_temp = 1              # NOT WORKING WHEN W_TEMP IS COMPUTED PROPERLY
        res_tex += images_0[cam][int(coords[1])][int(coords[0])] * w_temp
        w += w_temp
    if w != 0:
        return res_tex / w, True
    return res_tex, False
    
    
def show_texture():
    points = []
    colors = []

    for i in range(mesh_0.faces.shape[0]):
        A, B, C = vertices_0[i]
        for u in range(reso_0):
            for v in range(0, reso_0 - u):
                X = A + u / reso_0 * (B - A) + v / reso_0 * (C - A)
                tex_val, flag = getTex(i, X, u / reso_0, v / reso_0)
                if flag:
                    points.append(A + u / reso_0 * (B - A) + v / reso_0 * (C - A))
                    # colors.append(texture[i][u * reso + v] / wt[i][u * reso + v])
                    colors.append(tex_val)
    pcd = trimesh.PointCloud(vertices=points, colors=colors)
    pcd.scene().show()
#########################################################################################################################
#Fetch initial scene data
P,R,camera_locations,K,files = return_camera_info("camera_params.test","images.test")
mesh = PlyData.read('./scene_dense_mesh_refine.ply')
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

#Visibility table
Camera_visibility = calculate_visibility_mesh(camera_locations)
###########################################
### TEXTURE STUFF
start_0 = time()
CamCenter = camera_locations
image_files = files
mesh_0 = trimesh.load('scene_dense_mesh_refine.ply')
centroids_0 = mesh_0.vertices[mesh_0.faces].mean(axis=1)
vertices_0 = mesh_0.vertices[mesh_0.faces]
normals_0 = np.cross(vertices_0[:, 2] - vertices_0[:, 0], vertices_0[:, 1] - vertices_0[:, 0])
images_0 = Images
reso_0 = 5
visibility_0 =  Camera_visibility
texture_coords_0, b_img_tex_0 = init_texture_coords()
print("Time taken: ", time() - start_0)
show_texture()
###########################################
integeration,Energy_over_mesh = Energy_function_calc()
print("Total photometric loss without texture added:{}".format(integeration))

grad = Numerical_gradient_mesh(35)
print("Gradient:{}".format(grad))

print("Saving into PLY file....")
write_to_PLY(Vertex,Energy_over_mesh)
