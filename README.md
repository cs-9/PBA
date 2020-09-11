# Photometric Bundle Adjustment
This repository tries to implement the paper [Photometric Bundle Adjustment for Dense Multi-View 3D Modeling](https://hal.archives-ouvertes.fr/hal-00985811/document). This work was done by [Vishwesh Ramanathan](https://github.com/Vishwesh4) and [Gautam Biju Krishna](https://github.com/cs-9) as a course project in the course Geometry and Photometry in Computer vision at IIT Madras. Overall this repository uses OpenMVG, OpenMVS and python on the dinoRing and templeRing dataset. Please get back to us incase of any bugs. Additionally there are some simplifications used due to lack of computing power, overall the paper was implemented to the best of its knowledge. In order to view PLY files, please use Meshlab.

## Introduction
This paper focuses on jointly refining shape and camera parameters by minimizing photometric reprojec-
tion error between generated model images and observed images. This minimization is performed using a
gradient descent on the energy function accounting for visibility changes and optimizes for both shape and
camera parameters. This paper is used as a last refinement step in the 3D reconstruction pipeline.
The energy function we minimize is:  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/pastedimage_0.png)  

![Alttext](https://raw.github.com/cs-9/PBA/master/images/pastedimage1.png)

### Completed tasks

- [x] Get the SFM to work OpenMVG used
- [x] Get the Meshing to work OpenMVS used
- [x] Generate visibility function with assumption of nearby meshes to have same visibility
- [x] Texture estimation using simple weighting function 
- [x] Energy function calculation
- [x] Numerical Gradient calculation over X
- [x] Reflect changes in vertex over PLY file
- [x] Numerical Gradient calculation over camera parameters
- [x] Compilation of everything to a single pipeline
- [x] Try it on dinoRing dataset and templeRing dataset 

## Methodology
To solve this particular problem and to get a better scene reconstruction using photometric bundle adjust-
ments, the following methodology was followed. This section describes the overall overview of the method-
ology. The next section shines more light on the intermediate steps.
1. The first step involved the initial scene Ω construction using Structure from motion(SFM). This
initialization from SFM also gives us the initial camera calibration Π and initial point cloud X . This
construction is done using OpenMVG software. In order to deal with the surface, we need to form a
mesh. Using this point cloud a surface is constructed using triangular meshes. We used OpenMVS for
this process of mesh reconstruction.    
In order to run OpenMVG, we can use `SfM_SequentialPipeline.py` after installation   
`python SfM_SequentialPipeline.py {DATASET LOCATION} {RESULTS LOCATION}`    
Often there might be error in getting SFM results, we can rectify it by changing focal length. Please follow documentation of OpenMVG and OpenMVS.  
2. Since our photometric loss depends on the visibility of mesh from each camera, we calculate visibility
of each of the mesh from a particular camera position. Here its assumed each point in the mesh will
have same visibility as that of the mesh. A mesh is termed as visible if the ray from the camera center connecting to the centroid of the mesh does not encounter any other mesh. We can determine if a line segment intersects a triangle by comparing the signed volumes formed by different sections. Using this methodology, we successfully calculated the visibility of a single mesh from a single camera. Now this step was repeated for all the meshes and cameras.  
3. We next estimate the texture using the images from each camera and the point cloud. The texture at each point is computed using the following equation.  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/texture.png)  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/texture2.png)  
4. Now using all the terms we use the total photometric loss to form the energy function
5. Using the gradients over point cloud X and camera parameters Π , we use gradient descent over it to
minimize the energy function/photometric loss.

## Results
- <ins>SFM and MVS results</ins>  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/SFM_pipeline.png)
  
- <ins>Visibility function results</ins>  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/visibility.png)  
  
- <ins>Texture generation pipeline</ins>  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/texture_gen_pipeline.png)  
   
- <ins> Before and After PBA Energy function realization </ins>  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/photometric_visualization.png)  
 
- <ins> Loss results </ins>  
![Alttext](https://raw.github.com/cs-9/PBA/master/images/results.png)  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Note that your local environment should be cuda enabled from both pose detection and pose transfer.

### Prerequisites

Libraries/Software needed

```
plyfile
trimesh
OpenMVG[2]
OpenMVS[3]
```
### Running the files
* `1-datasets` - Contains templeRing, dinoRing and Chatteux dataset
* `2-SFM_results_datasets` - Contains ply file taken from SFM and OpenMVS results. The files camera_parames.test and images.test contains camera parameters R,C and P for templeRing dataset. These can be obtained from results from OpenMVS. Please ensure that they are in the same format
* `3-main-pipeline.py` - Contains main python file that performs PBA give mesh information. It outputs PLY file after doing bundle adjustments
* `4-results` - This folder contains all PLY results for both the datasets

## Reference
- [Amaël Delaunoy, Marc Pollefeys. Photometric Bundle Adjustment for Dense Multi-View 3D Modeling.2014 hal-00985811](https://hal.archives-ouvertes.fr/hal-00985811/document)
- [OpenMVG](https://github.com/openMVG/openMVG)
- [OpenMVS](https://github.com/cdcseacave/openMVS)