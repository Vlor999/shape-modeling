
import matplotlib.pyplot as plt
import trimesh

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Simple display for 3D meshes.')
    parser.add_argument('file', metavar='mesh_file', type=str, help='the mesh to display')

    args = parser.parse_args()

    mesh = trimesh.load(args.file)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:,1], mesh.vertices[:,2],
                    triangles=mesh.faces, linewidth=0.2, antialiased=True)

    plt.show()
