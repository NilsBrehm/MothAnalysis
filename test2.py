import pymuvr
from IPython import embed

observations_1 = [[[1.0, 2.3], # 1st observation, 1st cell
                    [0.2, 2.5, 2.7]],            # 2nd cell
                   [[1.1, 1.2, 3.0], # 2nd observation
                    []],
                   [[5.0, 7.8],
                    [4.2, 6.0]]]

cos = 0.1
tau = 1.0

a = pymuvr.square_dissimilarity_matrix(observations_1, cos, tau, 'distance')
embed()
exit()