from time import time
import numpy as np
from skimage import measure
import mlx.nn as nn
import trimesh
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Training
MAX_EPOCH = 20
BATCH_SIZE = 2<<14
print(f"Batch size: {BATCH_SIZE}")
resolution = 300
step = 2 / resolution

# MLP class
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 60),
            nn.Tanh(),
            nn.Linear(60, 120),
            nn.ReLU(),
            nn.Linear(120, 240),
            nn.GELU(),
            nn.Linear(240, 150),
            nn.ReLU(),
            nn.Linear(150, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
        )

    def __call__(self, x):
        return self.layers(x)

def loss_fn(model, X, y, weights):
    pred = model(X)
    loss = nn.losses.binary_cross_entropy(pred, y, with_logits=True)
    weighted_loss = mx.where(y == 1, loss * weights, loss)
    return mx.mean(weighted_loss)

def train_step(mlp, optimizer, batch_x, batch_y, p_weight):
    loss_grad = nn.value_and_grad(mlp, loss_fn)
    loss, grad = loss_grad(mlp, batch_x, batch_y, p_weight)
    optimizer.update(mlp, grad)
    return loss

def binary_acc(y_pred, y_test):
    y_pred_tag = mx.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = mx.round(accuracy * 100)
    return accuracy

def nif_train(data_in, data_out, batch_size):
    mlp = MLP()
    mx.eval(mlp.parameters())
    n_one = mx.sum(data_out == 1).item()
    p_weight = (data_out.shape[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)
    optimizer = optim.Adam(learning_rate=1e-3)

    for epoch in range(MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        permutation = mx.random.permutation(data_in.shape[0])
        data_in = data_in[permutation]
        data_out = data_out[permutation]
        current_loss = 0.

        for i in range(0, data_in.shape[0], batch_size):
            batch_x = data_in[i : i + batch_size]
            batch_y = data_out[i : i + batch_size]

            loss = train_step(mlp, optimizer, batch_x, batch_y, p_weight)

            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.5f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        mx.eval(mlp.parameters(), optimizer.state)
        avg_loss = current_loss / (data_in.shape[0] // batch_size)
        print(f"Epoch {epoch+1}/{MAX_EPOCH} - Loss: {avg_loss:.5f}")

    return mlp

def predict_grid(model, data_in, chunk_size=100000):
    outputs = []
    for i in range(0, data_in.shape[0], chunk_size):
        chunk = data_in[i : i + chunk_size]
        out = mx.sigmoid(model(chunk))
        mx.eval(out)
        outputs.append(np.array(out))
    return np.concatenate(outputs, axis=0)

def main():
    grid = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]
    X, Y, Z = grid
    data_in = mx.array(np.stack([X, Y, Z], -1).reshape(-1, 3))
    
    try:
        occupancy = mx.load("occupancy.npy")
        if type(occupancy) != mx.array:
            exit(0)
    except:
        occupancy = mx.ones((data_in.shape[0], 1))
    
    data_out = mx.reshape(occupancy, (-1, 1))

    mlp = nif_train(data_in, data_out, BATCH_SIZE)
    flat_params = tree_flatten(mlp.parameters())
    mx.savez("MLP_mlx.npz", **dict(flat_params))

    print("Génération de la grille finale...")
    occ = predict_grid(mlp, data_in)

    newocc = occ.reshape(resolution, resolution, resolution // 2)
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.5)
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit_mlx.off')
    print("Fichier alimplicit_mlx.off généré.")


if __name__ == "__main__":
    tic = time()
    main()
    tac = time()
    print(tac - tic) # batch 4096 -> 28s
