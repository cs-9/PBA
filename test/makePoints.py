radius = 1.0

import numpy as np

pi = np.pi

with open("sphericalPointCloud.txt", "w") as f:
    ALPHA = 25
    BETA = 9
    for i in range(ALPHA + 1):
        phi = i * pi / ALPHA
        r = radius * np.sin(phi)
        z = radius * np.cos(phi)
        for j in range(int(np.ceil(r * BETA))):
            theta = 2 * j * pi / BETA + np.random.uniform(0, pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            f.write("%f %f %f %f %f %f\n" % (x, y, z, x, y, z))

print("Done!")

