import matplotlib.image as mpimg
import numpy as np
from skimage import measure
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])


# Build 3D grids
# 3D Grids are of size: resolution x resolution x resolution/2
resolution = 300
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)
nl, nc, nz = occupancy.shape

# ---------- MAIN ----------
if __name__ == "__main__":
    
    images = []
    matricies = []
    for i in range(12):
        myFile = "image{0}.pgm".format(i)
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        images.append(img)
        calib_matrix = np.array(calib[i])
        calib_matrix = calib_matrix.reshape((3,4))
        matricies.append(calib_matrix)

    for l in range(nl):
        for c in range(nc):
            for z in range(nz):
                point = np.array([[X[l,c,z], Y[l,c,z], Z[l,c,z], 1]])
                for i, calib_mat in enumerate(matricies):
                    img = images[i]
                    projected_point = calib_mat @ point.T
                    x1, x2, x3 = projected_point[:, 0]
                    if x3 == 0: 
                        continue
                    u, v = x1 / x3, x2 / x3
                    if 0 <= v < img.shape[1] and 0 <= u < img.shape[0]:
                        if img[int(u), int(v)] == 0:
                            occupancy[l, c, z] = 0
                            break
                    else:
                        occupancy[l, c, z] = 0
                        break
        print(f"line: {l + 1}/{nl}")

    # Use the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)
    np.save("occupancy.npy", occupancy)

    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')