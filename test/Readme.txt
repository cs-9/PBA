Run makePoints.py to generate a point cloud for a sphere

Generate the output file sphericalPointCloud.txt
The format of the output is x, y, z, nx, ny, nz in each line
where (x, y, z) is the position of the point in space and (nx, ny, nz) is the normal to the point

Run as:
	python makePoints.py


Run disp3D.py to display the set of points as well the final mesh
Input must follow the previous format
install dependency open3d to run this

pip install open3d

Run as:
	python disp3D.py


