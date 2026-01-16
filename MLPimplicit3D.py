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
NUM_RANDOM_POINTS = 10_000_000

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

# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 60),
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
def nif_train(data_in, data_out, batch_size):
    # Initialize the MLP
    mlp = MLP()
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

        # Training is complete.
    print('MLP trained.')
    return mlp

def random_coordinates(nb_points: int, dimensions: list[tuple[float | int, float | int]]) -> list[np.ndarray]:
    """
    Generate the random coordinates
    
    Args:
        - nb_points (int): Number of points for each list
        - dimensions (list[tuple[int, int]]): the dimensions between we can found data
    
    Return:
        - X_1, X_2, ..., X_n: A list of len(dimensions) elements of random points
    """
    elements = []
    for dim in dimensions:
        start, stop = dim
        Xs = np.random.uniform(start, stop, nb_points)
        elements.append(Xs)
    return elements

def occupancy_random_points(X: np.ndarray, Y:np.ndarray, Z:np.ndarray, occupancy:np.ndarray, resolution: int) -> np.ndarray:
    """
    Convert relative positions:
    X [-1, 1] -> [0, resolution - 1]
    Y [-1, 1] -> [0, resolution - 1]
    Z [-0.5, 0.5] -> [0, resolution // 2 - 1]
    """
    x_coord = ((X + 1) * (resolution - 1) / 2.).astype(np.int64)
    y_coord = ((Y + 1) * (resolution - 1) / 2.).astype(np.int64)
    z_coord = ((Z + 0.5) * (resolution // 2 - 1)).astype(np.int64)
    x_coord = np.clip(x_coord, 0, resolution - 1)
    y_coord = np.clip(y_coord, 0, resolution - 1)
    z_coord = np.clip(z_coord, 0, resolution // 2 - 1)
    occupancy_rand = occupancy[x_coord, y_coord, z_coord]
    return occupancy_rand

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy

def main():
    Xrand, Yrand, Zrand = random_coordinates(NUM_RANDOM_POINTS, [(-1,1), (-1,1), (-0.5, 0.5)])
    # Format data for PyTorch
    data_in = np.stack((X, Y, Z), axis=-1)
    resolution_cube = resolution * resolution * resolution
    data_in = np.reshape(data_in, (resolution_cube // 2, 3))
    data_out = np.reshape(occupancy, (resolution_cube // 2, 1))

    data_in_rand = np.stack((Xrand, Yrand, Zrand), axis=-1)
    data_in_rand = np.reshape(data_in_rand, (NUM_RANDOM_POINTS, 3))

    data_out_rand = occupancy_random_points(Xrand, Yrand, Zrand, occupancy, resolution)
    data_out_rand = np.reshape(data_out_rand, (NUM_RANDOM_POINTS, 1))

    # Pytorch format
    data_in = torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    # Pytorch format
    data_in_rand = torch.from_numpy(data_in_rand)
    data_out_rand = torch.from_numpy(data_out_rand)

    # Train mlp
    mlp = nif_train(data_in_rand, data_out_rand, BATCH_SIZE)
    torch.save(mlp, "MLP.pt")

    # Visualization on training data
    outputs = mlp(data_in.float())
    occ = outputs.detach().cpu().numpy()

    # Go back to 3D grid
    newocc = np.reshape(occ, (resolution, resolution, resolution // 2))
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
