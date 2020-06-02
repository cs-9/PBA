radius = 1.0

import numpy as np

pi = np.pi

with open("sphericalPointCloud.txt", "w") as f:
    for i in range(51):
        phi = i * pi / 50
        r = radius * np.sin(phi)
        z = radius * np.cos(phi)
        for j in range(int(np.ceil(r * 12))):
            theta = j * pi / 6 + np.random.uniform(0, pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            f.write("%f %f %f %f %f %f\n" % (x, y, z, x, y, z))

print("Done!")

