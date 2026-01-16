from scipy.ndimage import map_coordinates
from sys import exit
from time import time
import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh

# Camera Calibration for Al's image[1..12].pgm
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
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

# Training
MAX_EPOCH = 10
BATCH_SIZE = 256
NUM_RANDOM_POINTS = 1_000_000

# Build 3D grids
# 3D Grids are of size resolution x resolution x resolution/2
resolution = 300
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# # Voxel occupancy
# occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# # Voxels are initially occupied then carved with silhouette information
# occupancy.fill(1)
occupancy = np.load("occupancy.npy")

def reduce_outside_occupancy(occ: np.ndarray, threshold: int | float = 1) -> tuple[np.ndarray, dict]:
    """Reduce the number of outside occupancy.
    
    To do so we are going to look if some layers can be erase:
        - if a plan(xy, xz, yz) is full of 0 and is a start we can erase it
    
    Args:
        - occ (np.ndarray): represente if a pixel is here or not
    
    Return:
        - np.ndarray: must represent the same object but with smaller dimensions
        - dict: bounds info with keys 'x_bounds', 'y_bounds', 'z_bounds', 'original_shape'
    """
    nl, nc, nz = occ.shape
    start_x, end_x = 0, nl
    start_y, end_y = 0, nc
    start_z, end_z = 0, nz
    
    for z in range(nz):
        plan_xy = occ[:, :, z]
        value = np.sum(plan_xy)
        if value > threshold:
            start_z = z
            break
    for z in range(nz - 1, -1, -1):
        plan_xy = occ[:, :, z]
        value = np.sum(plan_xy)
        if value > threshold:
            end_z = z
            break

    for y in range(nl):
        plan_xz = occ[:, y, :]
        value = np.sum(plan_xz)
        if value > threshold:
            start_y = y
            break
    for y in range(nl - 1, -1, -1):
        plan_xz = occ[:, y, :]
        value = np.sum(plan_xz)
        if value > threshold:
            end_y = y
            break

    for x in range(nc):
        plan_yz = occ[x, :, :]
        value = np.sum(plan_yz)
        if value > threshold:
            start_x = x
            break
    for x in range(nc - 1, -1, -1):
        plan_yz = occ[x, :, :]
        value = np.sum(plan_yz)
        if value > threshold:
            end_x = x
            break
    
    updated_occ = occ[start_x:end_x + 1, start_y:end_y + 1, start_z:end_z + 1]
    nb_voxels = np.prod(occ.shape)
    new_nb_voxels = np.prod(updated_occ.shape)
    print(f"deleted pixels: {100 * (nb_voxels - new_nb_voxels) / nb_voxels:3f}%")
    
    # Compute the new bounds in normalized coordinates
    orig_nl, orig_nc, orig_nz = occ.shape
    bounds = {
        'x_bounds': (-1 + 2 * start_x / orig_nl, -1 + 2 * (end_x + 1) / orig_nl),
        'y_bounds': (-1 + 2 * start_y / orig_nc, -1 + 2 * (end_y + 1) / orig_nc),
        'z_bounds': (-0.5 + start_z / orig_nz, -0.5 + (end_z + 1) / orig_nz),
        'original_shape': occ.shape,
        'new_shape': updated_occ.shape
    }
    return updated_occ, bounds

occupancy, occ_bounds = reduce_outside_occupancy(occupancy, 0)

# Update resolution and coordinate grids based on cropped occupancy
new_res_x, new_res_y, new_res_z = occupancy.shape
x_min, x_max = occ_bounds['x_bounds']
y_min, y_max = occ_bounds['y_bounds']
z_min, z_max = occ_bounds['z_bounds']

# Recreate coordinate grids for the cropped region
step_x = (x_max - x_min) / new_res_x
step_y = (y_max - y_min) / new_res_y
step_z = (z_max - z_min) / new_res_z
X, Y, Z = np.mgrid[x_min:x_max:step_x, y_min:y_max:step_y, z_min:z_max:step_z]

# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, L):
        super().__init__()
        self.L = L
        self.layers = nn.Sequential(
            nn.Linear(3 + 3 * 2 * L, 60), # each coordinated (3) is encoded according to the posiitonal encoding L * 2 (sin + cor) -> 3 * L * 2 + 3 (position)
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass """
        return self.layers(x)


# GPU or not GPU
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print("Device: ", device)

    
# MLP Training
def nif_train(data_in, data_out, batch_size, L):
    mlp = MLP(L)
    mlp = mlp.float()
    mlp.to(device)

    # Normalize cost between 0 and 1 in the grid
    n_one = (data_out == 1).sum()

    # loss for positives will be multiplied by this factor in the loss function
    p_weight = (data_out.size()[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()

    # sigmoid included in this loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight) # Pour éviter les descente de gradients trop forte
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    # Run the training loop
    for epoch in range(0, MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        # Creating batch indices
        permutation = torch.randperm(data_in.size()[0])

        # Set current loss value
        current_loss = 0.
        accuracy = 0

        # Iterate over batches
        for i in range(0, data_in.size()[0], batch_size):

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(batch_x.float())

            # Compute loss
            loss = loss_function(outputs, batch_y.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.5f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs, data_out)
        print("Binary accuracy: ", acc)

    print('MLP trained.')
    return mlp

def positional_encoding(coords: np.ndarray, L: int) -> np.ndarray:
    """
    Apply positional encoding to coordinates.
    
    For each coordinate x, computes:
    [x, sin(2^0 * π * x), cos(2^0 * π * x), ..., sin(2^(L-1) * π * x), cos(2^(L-1) * π * x)]
    
    Args:
        coords: array of shape (..., 3) containing x, y, z coordinates
        L: number of frequency bands
    
    Returns:
        array of shape (..., 3 + 3*2*L) with positional encoding
    """
    original_shape = coords.shape[:-1]
    coords_flat = coords.reshape(-1, 3)
    
    encoded = [coords_flat]
    for i in range(L):
        freq = (2 ** i) * np.pi
        encoded.append(np.sin(freq * coords_flat))
        encoded.append(np.cos(freq * coords_flat))
    result = np.concatenate(encoded, axis=-1)
    return result.reshape(*original_shape, -1)


def random_coordinates(nb_points: int, dimensions: list[tuple[float | int, float | int]]) -> np.ndarray:
    """
    Generate random 3D coordinates within given bounds.
    
    Args:
        - nb_points (int): Number of points to generate
        - dimensions (list[tuple]): List of (min, max) bounds for each dimension [x, y, z]
    
    Return:
        - np.ndarray of shape (nb_points, 3) with random coordinates
    """
    coords = np.zeros((nb_points, 3))
    for i, (start, stop) in enumerate(dimensions):
        coords[:, i] = np.random.uniform(start, stop, nb_points)
    return coords

def occupancy_random_points(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                            occupancy: np.ndarray, bounds: dict) -> np.ndarray:
    """
    Convert relative positions to occupancy grid indices.
    Points outside the bounds are considered unoccupied (0).
    
    Args:
        X, Y, Z: coordinate arrays
        occupancy: the cropped occupancy grid
        bounds: dict with 'x_bounds', 'y_bounds', 'z_bounds' keys
    """
    res_x, res_y, res_z = occupancy.shape
    x_min, x_max = bounds['x_bounds']
    y_min, y_max = bounds['y_bounds']
    z_min, z_max = bounds['z_bounds']
    
    # Normalize coordinates to [0, 1] within bounds, then scale to grid indices
    x_coord = ((X - x_min) / (x_max - x_min) * (res_x - 1)).astype(np.int64)
    y_coord = ((Y - y_min) / (y_max - y_min) * (res_y - 1)).astype(np.int64)
    z_coord = ((Z - z_min) / (z_max - z_min) * (res_z - 1)).astype(np.int64)
    
    inside_mask = (
        (X >= x_min) & (X < x_max) &
        (Y >= y_min) & (Y < y_max) &
        (Z >= z_min) & (Z < z_max)
    )
    

    x_coord = np.clip(x_coord, 0, res_x - 1)
    y_coord = np.clip(y_coord, 0, res_y - 1)
    z_coord = np.clip(z_coord, 0, res_z - 1)
    
    occupancy_rand = occupancy[x_coord, y_coord, z_coord]
    occupancy_rand = occupancy_rand * inside_mask
    return occupancy_rand

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy

def main():
    L = 10
    
    coords_rand = random_coordinates(NUM_RANDOM_POINTS, [
        occ_bounds['x_bounds'], 
        occ_bounds['y_bounds'], 
        occ_bounds['z_bounds']
    ])
    
    total_voxels = np.prod(occupancy.shape)
    coords_grid = np.stack((X, Y, Z), axis=-1)
    coords_grid = coords_grid.reshape(total_voxels, 3)
    
    data_in = positional_encoding(coords_grid, L)
    data_out = np.reshape(occupancy, (total_voxels, 1))

    data_in_rand = positional_encoding(coords_rand, L)
    
    Xrand, Yrand, Zrand = coords_rand[:, 0], coords_rand[:, 1], coords_rand[:, 2]
    data_out_rand = occupancy_random_points(Xrand, Yrand, Zrand, occupancy, occ_bounds)
    data_out_rand = np.reshape(data_out_rand, (NUM_RANDOM_POINTS, 1))

    data_in = torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    data_in_rand = torch.from_numpy(data_in_rand)
    data_out_rand = torch.from_numpy(data_out_rand)

    mlp = nif_train(data_in_rand, data_out_rand, BATCH_SIZE, L)
    torch.save(mlp, "MLP.pt")
    
    outputs = mlp(data_in.float())
    occ = outputs.detach().cpu().numpy()

    newocc = np.reshape(occ, occupancy.shape)
    newocc = np.around(newocc)

    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25)
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit.off')

# --------- MAIN ---------
if __name__ == "__main__":
    tic = time()
    main()
    tac = time()
    print(tac - tic) # 4m 27
