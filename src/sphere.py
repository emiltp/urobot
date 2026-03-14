import numpy as np
from scipy import linalg




def lstfit(points, plot=False):
    # image_points = [out.TransformPhysicalPointToIndex([point[0], point[1], point[2]]) for point in physical_points]

    # Setup and solve linear equation system.
    A = np.ones((len(points), 4))
    b = np.zeros(len(points))

    for row, point in enumerate(points):
        A[row, 0:3] = -2 * np.array(point)
        b[row] = -linalg.norm(point) ** 2

    res, _, _, _ = linalg.lstsq(A, b)


    print("The sphere's location is: {0:.2f}, {1:.2f}, {2:.2f}".format(*res[0:3]))
    print("The sphere's radius is: {0:.2f}mm".format(np.sqrt(linalg.norm(res[0:3]) ** 2 - res[3])))

    center = res[:3]
    radius = np.sqrt(linalg.norm(res[0:3]) ** 2 - res[3])

    
    if plot: plot3d(radius, center[0], center[1], center[2], points)

    return radius, center

def plot3d(r, x0, y0, z0, points):
    from matplotlib import rcParams
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    rcParams['font.family'] = 'serif'
    #   3D plot of the

    correctX = np.array(points)[:, 0]
    correctY = np.array(points)[:, 1]
    correctZ = np.array(points)[:, 2]

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v) * r
    y = np.sin(u) * np.sin(v) * r
    z = np.cos(v) * r
    x = x + x0
    y = y + y0
    z = z + z0

    #   3D plot of Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(correctX, correctY, correctZ, zdir='z', s=20, c='b', rasterized=True)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_aspect('auto')
    #ax.set_xlim3d(-35, 35)
    #ax.set_ylim3d(-35, 35)
    #ax.set_zlim3d(-70, 0)
    ax.set_xlabel('$x$ (mm)', fontsize=16)
    ax.set_ylabel('\n$y$ (mm)', fontsize=16)
    zlabel = ax.set_zlabel('\n$z$ (mm)', fontsize=16)
    plt.show()
    #plt.savefig('steelBallFitted.pdf', format='pdf', dpi=300, bbox_extra_artists=[zlabel], bbox_inches='tight')



