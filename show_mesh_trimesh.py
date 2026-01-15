
import trimesh

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Simple display for 3D meshes.')
    parser.add_argument('file', metavar='mesh_file', type=str, help='the mesh to display')

    args = parser.parse_args()

    mesh = trimesh.load(args.file)

    mesh.show()
