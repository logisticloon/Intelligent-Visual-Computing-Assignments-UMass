import argparse
import numpy as np
from skimage import measure
from sklearn.neighbors import KDTree
import open3d as o3d


def createGrid(points, resolution=96):
    """
    constructs a 3D grid containing the point cloud
    each grid point will store the implicit function value
    Args:
        points: 3D points of the point cloud
        resolution: grid resolution i.e., grid will be NxNxN where N=resolution
                    set N=16 for quick debugging, use *N=64* for reporting results
    Returns: 
        X,Y,Z coordinates of grid vertices     
        max and min dimensions of the bounding box of the point cloud                 
    """
    max_dimensions = np.max(
        points, axis=0)  # largest x, largest y, largest z coordinates among all surface points
    # smallest x, smallest y, smallest z coordinates among all surface points
    min_dimensions = np.min(points, axis=0)
    # com6pute the bounding box dimensions of the point cloud
    bounding_box_dimensions = max_dimensions - min_dimensions
    # extend bounding box to fit surface (if it slightly extends beyond the point cloud)
    max_dimensions = max_dimensions + bounding_box_dimensions/10
    min_dimensions = min_dimensions - bounding_box_dimensions/10
    X, Y, Z = np.meshgrid(np.linspace(min_dimensions[0], max_dimensions[0], resolution),
                          np.linspace(
        min_dimensions[1], max_dimensions[1], resolution),
        np.linspace(min_dimensions[2], max_dimensions[2], resolution))

    return X, Y, Z, max_dimensions, min_dimensions


def sphere(center, R, X, Y, Z):
    """
    constructs an implicit function of a sphere sampled at grid coordinates X,Y,Z
    Args:
        center: 3D location of the sphere center
        R     : radius of the sphere
        X,Y,Z : coordinates of grid vertices                      
    Returns: 
        IF    : implicit function of the sphere sampled at the grid points
    """
    IF = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + \
        (Z - center[2]) ** 2 - R ** 2
    return IF


def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    """
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)

    # Create an empty triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    # Use mesh.vertex to access the vertices' attributes
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    # Use mesh.triangle to access the triangles' attributes
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


def mlsReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points 
    The method shows reconstructed mesh
    Args:
        points :  points of the point cloud
                normals:  normals of the point cloud
                X,Y,Z  :  coordinates of grid vertices 
    Returns:
        IF     :  implicit function sampled at the grid points
    """

    ################################################
    # <================START CODE<================>
    ################################################

    # Calculating estimate of beta using KDTree

    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    tree = KDTree(points)
    _, idx = tree.query(points, k=2)
    beta = 2*np.mean(_[:, 1:].reshape(-1))
    # beta estimate complete

    # replace this random implicit function with your MLS implementation!
    IF = np.random.rand(X.shape[0], X.shape[1], X.shape[2]) - 0.5

    shapeX = X.shape
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    tree = KDTree(points)

    _, idx = tree.query(Q, k=50)

    IF = IF.reshape(-1)
    for i, fiftyIndices in enumerate(idx):
        p = Q[i]
        pts = points[fiftyIndices]
        ppts = p - pts
        nrmls = normals[fiftyIndices]
        dip = np.sum(nrmls*ppts, axis=1)
        phii = np.linalg.norm(ppts, axis=1)
        phii = phii*phii
        phii = phii/(-(beta*beta))
        phii = np.exp(phii)
        IF[i] = np.sum(phii*dip, axis=0)/np.sum(phii, axis=0)

    IF = IF.reshape(shapeX)

    ################################################
    # <================END CODE<================>
    ################################################

    return IF


def naiveReconstruction(points, normals, X, Y, Z):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    signed distance to the tangent plane of the surface point nearest to each 
    point (x,y,z)
    Args:
        points :  points of the point cloud
                normals:  normals of the point cloud
                X,Y,Z  :  coordinates of grid vertices 
    Returns:
        IF     : implicit function sampled at the grid points
    """

    ################################################
    # <================START CODE<================>
    ################################################

    # replace this random implicit function with your naive surface reconstruction implementation!
    IF = np.random.rand(X.shape[0], X.shape[1], X.shape[2]) - 0.5
    import time

    shapeX = X.shape
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    tree = KDTree(points)

    _, idx = tree.query(Q, k=1)

    pts = points[idx]
    nrmls = normals[idx]
    IF = IF.reshape(-1)
    for i, pointQ in enumerate(Q):
        IF[i] = nrmls[i]@(pointQ-pts[i])[0]
    IF = IF.reshape(shapeX)

    ################################################
    # <================END CODE<================>
    ################################################

    return IF


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Basic surface reconstruction')
    parser.add_argument('--file', type=str, default="sphere.pts",
                        help='input point cloud filename')
    parser.add_argument('--method', type=str, default="sphere",
                        help='method to use: mls (Moving Least Squares), naive (naive reconstruction), sphere (just shows a sphere)')
    args = parser.parse_args()

    # load the point cloud
    data = np.loadtxt(args.file)

    points = data[:, :3]
    normals = data[:, 3:6]

    # create grid whose vertices will be used to sample the implicit function
    X, Y, Z, max_dimensions, min_dimensions = createGrid(points, 64)

    if args.method == 'mls':
        print(f'Running Moving Least Squares reconstruction on {args.file}')
        IF = mlsReconstruction(points, normals, X, Y, Z)
    elif args.method == 'naive':
        print(f'Running naive reconstruction on {args.file}')
        IF = naiveReconstruction(points, normals, X, Y, Z)
    else:
        # toy implicit function of a sphere - replace this code with the correct
        # implicit function based on your input point cloud!!!
        print(f'Replacing point cloud {args.file} with a sphere!')
        center = (max_dimensions + min_dimensions) / 2
        R = max(max_dimensions - min_dimensions) / 4
        IF = sphere(center, R, X, Y, Z)

    showMeshReconstruction(IF)
