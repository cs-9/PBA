import numpy as np
import open3d as o3d

with open("pointCloud.txt", "r") as f:
    lines = f.readlines()
    pts = [line[:-1].split(" ") for line in lines]
    pts = np.array([[float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3]), float(pt[4]), float(pt[5])] for pt in pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(pts[:, 3:6])
    # Before
    o3d.visualization.draw_geometries([pcd])
    # After
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    poisson_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    o3d.visualization.draw_geometries([poisson_mesh], mesh_show_wireframe=True)

