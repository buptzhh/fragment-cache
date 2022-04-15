import csv

# with open("../data/175_future_and_perdition.csv", "a+", newline="", encoding='UTF-8') as out:
#     writer = csv.writer(out, delimiter=' ')
#     with open("../data/175_future.csv") as fu:
#         with open("../data/perdition.csv") as pe:
#             for line in fu:
#                 try:
#                     info = line.split(' ')
#                     info[3] = int(info[3])
#                     line2 = pe.readline()
#                     info.append(float(line2))
#                     writer.writerow(info)
#                 except ValueError:
#                     print(line)
#                     print(line2)
#                 pass

import numpy as np
shape = (5,9)
print(np.vstack((2 * np.random.random(shape) - 1, 2 * np.random.random((1, shape[1])) - 1)))