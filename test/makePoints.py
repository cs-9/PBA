import numpy as np
import sys

pi = np.pi

def sphere():
    radius = 1.0
    with open("pointCloud.txt", "w") as f:
        ALPHA = 25
        BETA = 9
        GAMMA = 0.05
        for i in range(ALPHA + 1):
            phi = i * pi / ALPHA
            r = radius * np.sin(phi)
            z = radius * np.cos(phi)
            for j in range(int(np.ceil(r * BETA))):
                theta = 2 * j * pi / BETA + np.random.uniform(0, pi)
                delta = np.random.uniform(0, GAMMA)
                x = r * np.cos(theta + delta)
                y = r * np.sin(theta + delta)
                f.write("%f %f %f %f %f %f\n" % (x, y, z, x, y, z))

def cube():
    side = 1.0
    def noise():
        return np.random.normal(0, side / 200)
        
    with open("pointCloud.txt", "w") as f:
        NUM_POINTS = 6
        # Fix z
        z = -side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                x = i * side / NUM_POINTS + noise() - side / 2
                y = j * side / NUM_POINTS + noise() - side / 2
                z = z + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 0, 0, -1))
        z = side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                x = i * side / NUM_POINTS + noise() - side / 2
                y = j * side / NUM_POINTS + noise() - side / 2
                z = z + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 0, 0, 1))
        # Fix y
        y = -side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                x = i * side / NUM_POINTS + noise() - side / 2
                z = j * side / NUM_POINTS + noise() - side / 2
                y = y + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 0, -1, 0))
        y = side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                x = i * side / NUM_POINTS + noise() - side / 2
                z = j * side / NUM_POINTS + noise() - side / 2
                y = y + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 0, 1, 0))
        # Fix x
        x = -side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                z = i * side / NUM_POINTS + noise() - side / 2
                y = j * side / NUM_POINTS + noise() - side / 2
                x = x + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, -1, 0, 0))
        x = side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                z = i * side / NUM_POINTS + noise() - side / 2
                y = j * side / NUM_POINTS + noise() - side / 2
                x = x + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 1, 0, 0))
                
def imbricated_cube():
    # A cube with a sphere in the center of the top face
    side = 1.0
    radius = np.sqrt(2) / 2
    center = 0.5
    def noise():
        # return np.random.normal(0, side / 250)
        return 0
    # Bottom face definitely of cube
    with open("pointCloud.txt", "w") as f:
        NUM_POINTS = 50
        z = -side / 2
        for i in range(NUM_POINTS):
            for j in range(NUM_POINTS):
                x = i * side / NUM_POINTS + noise() - side / 2
                y = j * side / NUM_POINTS + noise() - side / 2
                z = z + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, 0, 0, -1))

        # Rest of the cube
        for i in range(1, NUM_POINTS):
            z = i * side / NUM_POINTS - side / 2
            if center - radius < z and center + radius > z:
                # z = radius * cos (phi)
                phi = np.arccos(np.abs(z - center) / radius)
                r = radius * np.sin(phi)
                for j in range(4):
                    for i in range(NUM_POINTS):
                        c1 = c2 = nx = ny = 0
                        if j == 0:
                            c2 = i * side / NUM_POINTS
                            nx = -1
                        elif j == 1:
                            c1 = i * side / NUM_POINTS
                            ny = -1
                        elif j == 2:
                            c1 = side
                            c2 = i * side / NUM_POINTS
                            nx = 1
                        else:
                            c2 = side
                            c1 = i * side / NUM_POINTS
                            ny = 1
                        x = c1 + noise() - side / 2
                        y = c2 + noise() - side / 2
                        z = z + noise()
                        if x * x + y * y >= r * r:
                            f.write("%f %f %f %f %f %f\n" % (x, y, z, nx, ny, 0))
                        else:
                            # can be part of cube or sphere
                            cx = cy = 0
                            if j == 0 or j == 2:
                                # Fix y
                                cx = np.sqrt(r * r - y * y) + noise()
                                if cx > x:
                                    if j == 2:
                                        cx = -cx
                                    f.write("%f %f %f %f %f %f\n" % (cx, y, z, cx, y, z - center))
                                else: 
                                    f.write("%f %f %f %f %f %f\n" % (x, y, z, nx, ny, 0))
                            else:
                                # Fix x
                                cy = np.sqrt(r * r - x * x) + noise()
                                if cy > y:
                                    if j == 3:
                                        cy = -cy
                                    f.write("%f %f %f %f %f %f\n" % (x, cy, z, x, cy, z - center))
                                else:
                                    f.write("%f %f %f %f %f %f\n" % (x, y, z, nx, ny, 0))
                                
            else:
                for j in range(4):
                    for i in range(NUM_POINTS):
                        c1 = c2 = nx = ny = 0
                        if j == 0:
                            c2 = i * side / NUM_POINTS
                            nx = -1
                        elif j == 1:
                            c1 = i * side / NUM_POINTS
                            ny = -1
                        elif j == 2:
                            c1 = side
                            c2 = i * side / NUM_POINTS
                            nx = 1
                        else:
                            c2 = side
                            c1 = i * side / NUM_POINTS
                            ny = 1
                        x = c1 + noise() - side / 2
                        y = c2 + noise() - side / 2
                        z = z + noise()
                        f.write("%f %f %f %f %f %f\n" % (x, y, z, nx, ny, 0))
        
        NUM_POINTS2 = int(NUM_POINTS * radius / side)
        for i in range(NUM_POINTS2):
            z = (i + 1) * radius / NUM_POINTS2 + side / 2
            phi = np.arccos(np.abs(z - center) / radius)
            r = radius * np.sin(phi)
            for j in range(4 * NUM_POINTS2):
                theta = 2 * j * np.pi / NUM_POINTS2
                x = r * np.cos(theta) + noise()
                y = r * np.sin(theta) + noise()
                z = z + noise()
                f.write("%f %f %f %f %f %f\n" % (x, y, z, x, y, z - center))
                
            
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sphere()
    else:
        globals()[sys.argv[1]]()
    print("Done!")
