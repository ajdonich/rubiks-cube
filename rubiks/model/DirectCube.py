import math
import numpy as np

from rubiks.model.VectorCube import VectorCube, SIDES, WHITE_CB, W, ORANGE_CB, O, GREEN_CB, G, RED_CB, R, BLUE_CB, B, YELLOW_CB, Y

class DirectCube(VectorCube):
#{
    # Access through static methods below 
    _direction_matrix = None
    _direction_index  = None

    # Hardcoded-index arrays for optimization
    _dir_order_cn_ctr = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,
                                  56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,97,99,101,103,105,
                                  107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143])
    _dir_order_ed_ctr = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,
                                  57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,
                                  107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143])
    _dir_order_cn_ed  = np.array([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,
                                 56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,
                                 106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142])
    _dir_order_ed     = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,
                                  57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,96,98,100,102,104,
                                  106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142])

    # Lock hard indexes
    _dir_order_ed_ctr.flags.writeable = False
    _dir_order_cn_ctr.flags.writeable = False
    _dir_order_cn_ed.flags.writeable = False
    _dir_order_ed.flags.writeable = False

    @staticmethod
    def direction_matproto():
    #{
        # Creates prototype for direction/orientation matrix, column ordering corresponds with 
        # FaceletSolver's order_heuristic fcn calc (i.e. VectorCube's _order_cn and _order_ed)

        if DirectCube._direction_matrix is None:
        #{
            cned_flets  = VectorCube._facelet_matrix[:, VectorCube._order_cn_ed]
            edcnt_flets = VectorCube._facelet_matrix[:, VectorCube._order_ed_cnt]
            diff_vecs   = cned_flets[2:] - edcnt_flets[2:]

            tindex = 0
            dmatrix = np.zeros((5,144), dtype=int)
            for diffv, cned, edcnt in zip(diff_vecs.T , cned_flets.T, edcnt_flets.T):
            #{
                dmatrix[:2, tindex]   = cned[:2]
                dmatrix[2:, tindex]   = -diffv/2
                dmatrix[:2, tindex+1] = edcnt[:2]
                dmatrix[2:, tindex+1] = diffv/2
                tindex += 2
            #}

            DirectCube._direction_matrix = dmatrix
            DirectCube._direction_matrix.flags.writeable = False
        #}

        return DirectCube._direction_matrix
    #}

    @staticmethod
    def get_direction_index(cindex):
    #{
        # Each column of _direction_index contains UP TO THREE indices into _direction_matrix
        # per ONE _facelet_matrix index. Corner flets have 2 direction indexes, edges have 3
        # and centers have 4, thus some _direction_matrix values are left unassigned as == -1

        if DirectCube._direction_index is None:
        #{
            dindex = np.ones((6,54), dtype=int) * -1
            dindex[:2] = VectorCube._facelet_matrix[:2] 
            for i, flet in enumerate(VectorCube._facelet_matrix.T):
            #{
                row = 2
                for j, tface in enumerate(DirectCube.direction_matproto().T):
                    if sum(tface[:2] == flet[:2]) == 2:
                        dindex[row,i] = j
                        row += 1
            #}
            DirectCube._direction_index = dindex
            DirectCube._direction_index.flags.writeable = False
        #}

        # Here strip superfluous -1's from final returned direction index 
        flat_index = DirectCube._direction_index[2:, cindex].T.flatten()
        return flat_index[(flat_index != -1)]
    #}

    # Begin DirectCube implementation
    def __init__(self, copycube=None):
        super(DirectCube, self).__init__(copycube)
        if copycube is None: self.direction_matrix = DirectCube.direction_matproto().copy()
        else: self.direction_matrix = copycube.direction_matrix.copy()

    def state(self):
        return np.concatenate((self.facelet_matrix[2:].T.reshape(162,), self.direction_matrix[2:].T.reshape(432,)))

    def reset(self, state=None):
        if state is None: 
            np.copyto(self.facelet_matrix, VectorCube._facelet_matrix)
            np.copyto(self.direction_matrix, DirectCube.direction_matproto())
        else: 
            self.facelet_matrix[2:,:] = state[:162].reshape(54,3).T
            self.direction_matrix[2:,:] = state[162:].reshape(144,3).T
        return self

    def rotate(self, move):
    #{
        cindex = self.get_facelet_indices(move[0])
        dindex = DirectCube.get_direction_index(cindex)
        rmat = VectorCube.rotation(move)

        self.facelet_matrix[2:, cindex] = rmat.dot(self.facelet_matrix[2:, cindex])
        self.direction_matrix[2:, dindex] = rmat.dot(self.direction_matrix[2:, dindex])
        return self
    #}
#}