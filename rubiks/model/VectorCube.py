import math
import numpy as np

GODS_NUMBER = 20

# COLORS/SIDES:
WHITE_CB, W  = 1, 1
ORANGE_CB, O = 2, 2
GREEN_CB, G  = 3, 3
RED_CB, R    = 4, 4
BLUE_CB, B   = 5, 5
YELLOW_CB, Y = 6, 6

# COLORS/SIDES in order
SIDES = [WHITE_CB, ORANGE_CB, GREEN_CB, RED_CB, BLUE_CB, YELLOW_CB]

# Print convenience maps/fcns
color_letr_map = { 0:0, WHITE_CB:'W', ORANGE_CB:'O', GREEN_CB:'G', RED_CB:'R', BLUE_CB:'B', YELLOW_CB:'Y' }
color_name_map = { 0:'N/A', WHITE_CB:'WHITE', ORANGE_CB:'ORANGE', GREEN_CB:'GREEN', RED_CB:'RED', BLUE_CB:'BLUE', YELLOW_CB:'YELLOW' }

def color_letr(fc): return color_letr_map[fc]
def color_name(fc): return color_name_map[fc]
def tri_color_name(tc): return (f"({color_letr(tc[0])},{color_letr(tc[1])},{color_letr(tc[2])})")    

class VectorCube:
#{
    # The 18 possible moves from any cube state
    MOVES = [(sd,ang) for sd in SIDES for ang in [90,-90,180]]
    ACTIONS = {mv: ac for ac, mv in enumerate(MOVES)}

    # Access through static methods below
    _rotation_matrices = {}
    _rotation_angle = {}

    # Hardcoded-index arrays for optimization
    _centers      = np.array([4,13,22,31,40,49])
    _minimalindex = np.array([0,1,2,3,5,6,7,8,21,23,39,41,45,46,47,48,50,51,52,53])
    _movableindex = np.array([0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,23,24,25,26,27,
                              28,29,30,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,50,51,52,53])

    # Index arrays for 3-flet-bars and 8-flet-rings for all sides
    _cn_ring_prv  = np.array([0,2,8,6,9,11,17,15,18,20,26,24,27,29,35,33,36,38,44,42,45,47,53,51])
    _ed_ring_mid  = np.array([1,5,7,3,10,14,16,12,19,23,25,21,28,32,34,30,37,41,43,39,46,50,52,48])
    _cn_ring_nxt  = np.array([2,8,6,0,11,17,15,9,20,26,24,18,29,35,33,27,38,44,42,36,47,53,51,45])                        
    _order_cn_cnt = np.concatenate((_cn_ring_prv, _cn_ring_nxt, np.broadcast_to(_centers, (4,6)).T.flatten()))
    _order_ed_cnt = np.concatenate((_ed_ring_mid, _ed_ring_mid, np.broadcast_to(_centers, (4,6)).T.flatten()))
    _order_cn_ed  = np.concatenate((_cn_ring_prv, _cn_ring_nxt, _ed_ring_mid))
    _order_ed     = np.concatenate((_ed_ring_mid, _ed_ring_mid, _ed_ring_mid))

    # Prototype cube data structure
    _facelet_matrix = np.array\
        ([[1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6],
          [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
          [-2,-2,-2,0,0,0,2,2,2,-2,0,2,-2,0,2,-2,0,2,3,3,3,3,3,3,3,3,3,2,0,-2,2,0,-2,2,0,-2,-3,-3,-3,-3,-3,-3,-3,-3,-3,2,2,2,0,0,0,-2,-2,-2],
          [-2,0,2,-2,0,2,-2,0,2,-3,-3,-3,-3,-3,-3,-3,-3,-3,-2,0,2,-2,0,2,-2,0,2,3,3,3,3,3,3,3,3,3,2,0,-2,2,0,-2,2,0,-2,-2,0,2,-2,0,2,-2,0,2],
          [3,3,3,3,3,3,3,3,3,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,-3,-3,-3,-3,-3,-3,-3,-3,-3]])
    
    # Lock prototype and hard-coded indexes
    _centers.flags.writeable      = False
    _minimalindex.flags.writeable = False
    _movableindex.flags.writeable = False
    _cn_ring_prv.flags.writeable  = False
    _ed_ring_mid.flags.writeable  = False
    _cn_ring_nxt.flags.writeable  = False
    _order_cn_cnt.flags.writeable = False
    _order_ed_cnt.flags.writeable = False
    _order_cn_ed.flags.writeable  = False
    _order_ed.flags.writeable     = False
    _facelet_matrix.flags.writeable = False


    # Naked-eye _facelet_matrix index lookup chart
    # idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]
    # col: [1,1,1,1,1,1,1,1,1,2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    # ci:  [0,1,2,3,4,5,6,7,8,0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    @staticmethod
    def rotation(side_angle_tuple):
    #{
        if not VectorCube._rotation_matrices:
        #{
            # Rotation matrix mappings
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                VectorCube._rotation_matrices[(GREEN_CB,deg)]  = Rx
                VectorCube._rotation_matrices[(BLUE_CB,deg)]   = Rx
                VectorCube._rotation_matrices[(RED_CB,deg)]    = Ry
                VectorCube._rotation_matrices[(ORANGE_CB,deg)] = Ry
                VectorCube._rotation_matrices[(WHITE_CB,deg)]  = Rz
                VectorCube._rotation_matrices[(YELLOW_CB,deg)] = Rz
            #}
        #}
        
        return VectorCube._rotation_matrices[side_angle_tuple]
    #}
    
    @staticmethod
    def angle(rotation_matrix):
    #{
        if not VectorCube._rotation_angle:
        #{
            # Rotation matrix mappings
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                VectorCube._rotation_angle[tuple(Rx.flatten())] = deg
                VectorCube._rotation_angle[tuple(Ry.flatten())] = deg
                VectorCube._rotation_angle[tuple(Rz.flatten())] = deg
            #}
        #}
        
        return VectorCube._rotation_angle[tuple(rotation_matrix.flatten())]
    #}

    @staticmethod
    def inverse(move):
        return move if move[1] == 180 else (move[0], -move[1])

    @staticmethod
    def inverse_seq(moves):
        return [VectorCube.inverse(mv) for mv in reversed(moves)]
        
    @staticmethod
    def random_moves(sz=20):
        return [VectorCube.MOVES[action] for action in np.random.randint(len(VectorCube.MOVES), size=sz)]

    # Vector cube implementation
    def __init__(self, copycube=None):
        if copycube is None: self.facelet_matrix = VectorCube._facelet_matrix.copy()
        else: self.facelet_matrix = copycube.facelet_matrix.copy()

    def isinlayer(self, side):
        cent_pos = VectorCube._facelet_matrix[2:, VectorCube._centers[side-1]]
        return np.nonzero(self.facelet_matrix[2:,:].T.dot(cent_pos) > 0)[0]

    def isonside(self, side):
        cent_pos = VectorCube._facelet_matrix[2:, VectorCube._centers[side-1]]
        return np.nonzero(self.facelet_matrix[2:,:].T.dot(cent_pos) >= 9)[0]

    # Returns facelet_matrix col-index array
    def get_facelet_indices(self, side, layer=True):
        return self.isinlayer(side) if layer else self.isonside(side)

    # Called by CubeView class
    def get_mask_cis(self, flet_idx):
        return [VectorCube._facelet_matrix[:2, col] for pos in self.facelet_matrix[2:, flet_idx].T
                for col in range(VectorCube._facelet_matrix.shape[1]) if sum(VectorCube._facelet_matrix[2:,col] == pos) == 3]

    # Called by CubeView class
    def get_colors(self, side):
        side_index = np.nonzero(VectorCube._facelet_matrix[0] == side)[0]
        return [self.facelet_matrix[0, col] for pos in VectorCube._facelet_matrix[2:, side_index].T 
                for col in range(self.facelet_matrix.shape[1]) if sum(self.facelet_matrix[2:,col] == pos) == 3]

    def dot(self, flet_idx_pair):
        return self.facelet_matrix[2:, flet_idx_pair[0]].dot(self.facelet_matrix[2:, flet_idx_pair[1]])

    def compare(self, other=None, flet_index=None):
        if flet_index is None: flet_index = VectorCube._movableindex
        if other is not None: return (self.facelet_matrix[2:, flet_index] == other.facelet_matrix[2:, flet_index])
        else: return (self.facelet_matrix[2:, flet_index] == VectorCube._facelet_matrix[2:, flet_index])

    def equal(self, other, flet_index=None):
        logical = self.compare(other=other, flet_index=flet_index)
        return sum(sum(logical)) == (logical.shape[0] * logical.shape[1])

    def solved(self, flet_index=None):
        logical = self.compare(flet_index=flet_index)
        return sum(sum(logical)) == (logical.shape[0] * logical.shape[1])

    # Bit faster than solved(...) above
    def is_solved(self):
        return sum(sum(self.facelet_matrix[2:] == VectorCube._facelet_matrix[2:])) == 162

    def state(self):
        return self.facelet_matrix[2:].T.reshape(162,)

    def nn_state(self):
        return self.facelet_matrix[2:, VectorCube._movableindex].T.reshape(144,)
    
    def reset(self, state=None):
        if state is None: np.copyto(self.facelet_matrix, VectorCube._facelet_matrix)
        else: self.facelet_matrix[2:,:] = state.reshape(54,3).T
        return self
    
    def scramble(self, sz=64):
        self.reset() # Always scrambles from a solved/reset state
        for action in np.random.randint(len(VectorCube.MOVES), size=sz): 
            self.rotate(VectorCube.MOVES[action])
        return self

    def trace_scramble(self, sz=20, apply_moves=False):
    #{
        # Assures non-redundant sequence of moves (i.e. no-back-to-back-rotations-of-same-side) 
        cands = [VectorCube.MOVES[action] for action in np.random.randint(len(VectorCube.MOVES), size=(sz*2))]
        moves = [cands[i] for i in range(len(cands)) if (i == 0) or (cands[i-1][0] != cands[i][0])]
        if apply_moves: self.rotate_seq(moves[:sz])
        
        return moves[:sz], [self.inverse(mv) for mv in reversed(moves[:sz])]
    #}

    def rotate(self, move):
        cindex = self.get_facelet_indices(side=move[0])
        self.facelet_matrix[2:, cindex] = VectorCube.rotation \
            (move).dot(self.facelet_matrix[2:, cindex])
        return self
    
    def rotate_seq(self, path):
        for move in path: self.rotate(move)
        return self
#}
