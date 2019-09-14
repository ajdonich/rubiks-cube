import math
import numpy as np

from VectorCube import VectorCube, SIDES, WHITE_CB, W, ORANGE_CB, O, GREEN_CB, G, RED_CB, R, BLUE_CB, B, YELLOW_CB, Y

# GODS_NUMBER = 20

# # COLORS/SIDES:
# WHITE_CB, W  = 1, 1
# ORANGE_CB, O = 2, 2
# GREEN_CB, G  = 3, 3
# RED_CB, R    = 4, 4
# BLUE_CB, B   = 5, 5
# YELLOW_CB, Y = 6, 6

# # COLORS/SIDES in order
# SIDES = [WHITE_CB, ORANGE_CB, GREEN_CB, RED_CB, BLUE_CB, YELLOW_CB]

# # Print convenience maps/fcns
# color_letr_map = { 0:0, WHITE_CB:'W', ORANGE_CB:'O', GREEN_CB:'G', RED_CB:'R', BLUE_CB:'B', YELLOW_CB:'Y' }
# color_name_map = { 0:'N/A', WHITE_CB:'WHITE', ORANGE_CB:'ORANGE', GREEN_CB:'GREEN', RED_CB:'RED', BLUE_CB:'BLUE', YELLOW_CB:'YELLOW' }

# def color_letr(fc): return color_letr_map[fc]
# def color_name(fc): return color_name_map[fc]
# def tri_color_name(tc): return (f"({color_letr(tc[0])},{color_letr(tc[1])},{color_letr(tc[2])})")

class Opticube(VectorCube):
#{
    # Solved facelet position vectors, ordered left-to-right top-to-bottom per side
    SOLVED_POS = { WHITE_CB:   np.array([[-2,-2,3],[-2,0,3],[-2,2,3],[0,-2,3],[0,0,3],[0,2,3],[2,-2,3],[2,0,3],[2,2,3]]),
                   ORANGE_CB:  np.array([[-2,-3,2],[0,-3,2],[2,-3,2],[-2,-3,0],[0,-3,0],[2,-3,0],[-2,-3,-2],[0,-3,-2],[2,-3,-2]]),
                   GREEN_CB:   np.array([[3,-2,2],[3,0,2],[3,2,2],[3,-2,0],[3,0,0],[3,2,0],[3,-2,-2],[3,0,-2],[3,2,-2]]),
                   RED_CB:     np.array([[2,3,2],[0,3,2],[-2,3,2],[2,3,0],[0,3,0],[-2,3,0],[2,3,-2],[0,3,-2],[-2,3,-2]]),
                   BLUE_CB:    np.array([[-3,2,2],[-3,0,2],[-3,-2,2],[-3,2,0],[-3,0,0],[-3,-2,0],[-3,2,-2],[-3,0,-2],[-3,-2,-2]]),
                   YELLOW_CB:  np.array([[2,-2,-3],[2,0,-3],[2,2,-3],[0,-2,-3],[0,0,-3],[0,2,-3],[-2,-2,-3],[-2,0,-3],[-2,2,-3]]) }
    
    # The 18 possible moves from any cube state
    MOVES = [(sd,ang) for sd in SIDES for ang in [90,-90,180]]
    ACTIONS = {mv: ac for ac, mv in zip(range(18), MOVES)}

    # Access through static methods below
    _rotation_matrices = {}

    # Do NOT modify the following: hardcoded difference-index and solved  
    # position matrices for optimized cube creation and distance calculations
    _aindex = np.array([0,0,0,1,2,2,2,3,5,6,6,6,
                7,8,8,8,9,9,9,10,11,11,11,12,
                14,15,15,15,16,17,17,17,18,18,18,19,
                20,20,20,21,23,24,24,24,25,26,26,26,
                27,27,27,28,29,29,29,30,32,33,33,33,
                34,35,35,35,36,36,36,37,38,38,38,39,
                41,42,42,42,43,44,44,44,45,45,45,46,
                47,47,47,48,50,51,51,51,52,53,53,53])
    _aindex.flags.writeable = False

    _bindex = np.array([4,1,3,4,4,1,5,4,4,4,3,7,
                4,4,5,7,13,10,12,13,13,10,14,13,
                13,13,12,16,13,13,14,16,22,19,21,22,
                22,19,23,22,22,22,21,25,22,22,23,25,
                31,28,30,31,31,28,32,31,31,31,30,34,
                31,31,32,34,40,37,39,40,40,37,41,40,
                40,40,39,43,40,40,41,43,49,46,48,49,
                49,46,50,49,49,49,48,52,49,49,50,52])
    _bindex.flags.writeable = False

    _edindex = np.array([1,3,5,7,10,12,14,16,19,21,23,25,28,30,32,34,37,39,41,43,46,48,50,52])
    _edindex.flags.writeable = False

    _cnindex = np.array([0,2,6,8,9,11,15,17,18,20,24,26,27,29,33,35,36,38,42,44,45,47,51,53])
    _cnindex.flags.writeable = False
    
    _centers = np.array([4,13,22,31,40,49])
    _centers.flags.writeable = False

    _movableindex = np.array([0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,23,24,25,26,27,
                              28,29,30,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,50,51,52,53])
    _movableindex.flags.writeable = False
    
    _whtcrossindex = np.array([1,3,5,7,10,19,28,37])
    _whtcrossindex.flags.writeable = False

    _tla_index = np.array([1,3,5,6,7,10,11,18,19,28,37])
    _tlb_index = np.array([1,3,5,6,7,8,10,11,18,19,20,27,28,37])
    _tlc_index = np.array([1,2,3,5,6,7,8,10,11,18,19,20,27,28,29,36,37])
    _tlayerindex = np.array([0,1,2,3,5,6,7,8,9,10,11,18,19,20,27,28,29,36,37,38])

    _tla_index.flags.writeable = False
    _tlb_index.flags.writeable = False
    _tlc_index.flags.writeable = False
    _tlayerindex.flags.writeable = False
    
    _mlayerindex = np.array([12,14,21,23,30,32,39,41])
    _mlayerindex.flags.writeable = False

    _t2la_index = np.array([0,1,2,3,5,6,7,8,9,10,11,14,18,19,20,21,27,28,29,36,37,38])
    _t2lb_index = np.array([0,1,2,3,5,6,7,8,9,10,11,14,18,19,20,21,23,27,28,29,30,36,37,38])
    _t2lc_index = np.array([0,1,2,3,5,6,7,8,9,10,11,14,18,19,20,21,23,27,28,29,30,32,36,37,38,39])
    _t2l_index = np.array([0,1,2,3,5,6,7,8,9,10,11,12,14,18,19,20,21,23,27,28,29,30,32,36,37,38,39,41])

    _t2la_index.flags.writeable = False
    _t2lb_index.flags.writeable = False
    _t2lc_index.flags.writeable = False
    _t2l_index.flags.writeable = False

    _oll_index = np.concatenate((_t2l_index, [45, 46, 47, 48, 50, 51, 52, 53]))
    _oll_index.flags.writeable = False
    
    _blayerindex = np.array([15,16,17,24,25,26,33,34,35,42,43,44,45,46,47,48,50,51,52,53])
    _blayerindex.flags.writeable = False
    
    _facelet_matrix = np.array\
        ([[1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6],
          [0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8],
          [-2,-2,-2,0,0,0,2,2,2,-2,0,2,-2,0,2,-2,0,2,3,3,3,3,3,3,3,3,3,2,0,-2,2,0,-2,2,0,-2,-3,-3,-3,-3,-3,-3,-3,-3,-3,2,2,2,0,0,0,-2,-2,-2],
          [-2,0,2,-2,0,2,-2,0,2,-3,-3,-3,-3,-3,-3,-3,-3,-3,-2,0,2,-2,0,2,-2,0,2,3,3,3,3,3,3,3,3,3,2,0,-2,2,0,-2,2,0,-2,-2,0,2,-2,0,2,-2,0,2],
          [3,3,3,3,3,3,3,3,3,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,2,2,2,0,0,0,-2,-2,-2,-3,-3,-3,-3,-3,-3,-3,-3,-3]])
    _facelet_matrix.flags.writeable = False
    
    _DFACTOR = sum(np.linalg.norm(_facelet_matrix[2:, _bindex] - _facelet_matrix[2:, _aindex], axis=0))
    
    @staticmethod
    def rotation(side_angle_tuple):
    #{
        if not Opticube._rotation_matrices:
        #{
            # Rotation matrix mappings
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                Opticube._rotation_matrices[(GREEN_CB,deg)]  = Rx
                Opticube._rotation_matrices[(BLUE_CB,deg)]   = Rx
                Opticube._rotation_matrices[(RED_CB,deg)]    = Ry
                Opticube._rotation_matrices[(ORANGE_CB,deg)] = Ry
                Opticube._rotation_matrices[(WHITE_CB,deg)]  = Rz
                Opticube._rotation_matrices[(YELLOW_CB,deg)] = Rz
            #}
        #}
        
        return Opticube._rotation_matrices[side_angle_tuple]
    #}
    
    @staticmethod
    def inverse(move):
        return move if move[1] == 180 else (move[0], -move[1])
        
    @staticmethod
    def random_moves(sz=20):
        return [Opticube.MOVES[action] for action in np.random.randint(len(Opticube.MOVES), size=sz)]

    # Begin Opticube implementation
    def __init__(self, copycube=None):
        super(Opticube, self).__init__(copycube)
    
    def isinlayer(self, side):
        return np.argwhere(self.facelet_matrix[2:,:].T.dot(Opticube.SOLVED_POS[side][4]) > 0).flatten()

    def isonside(self, side):
        return np.argwhere(self.facelet_matrix[2:,:].T.dot(Opticube.SOLVED_POS[side][4]) >= 9).flatten()
    
    # Returns facelet_matrix col-index array
    def get_facelet_indices(self, side, layer=True):
        return self.isinlayer(side) if layer else self.isonside(side)
    
    # def get_solved_indices(self):
    #     return np.argwhere(self.facelet_matrix[2:,:].T.dot(Opticube.SOLVED_POS[side][4]) > 0).flatten()
    
    def get_edgelets(self):
        return self.facelet_matrix[:, Opticube._edindex]

    def get_cornerlets(self):
        return self.facelet_matrix[:, Opticube._cnindex]
    
    def get_movable_flets(self):
        return self.facelet_matrix[:, Opticube._movableindex]
    
    def get_layer_flets(self, layer):
        if layer == 't':   self.facelet_matrix[:, Opticube._tlayerindex]
        elif layer == 'm': self.facelet_matrix[:, Opticube._mlayerindex]
        elif layer == 'b': self.facelet_matrix[:, Opticube._blayerindex]
        
    def get_colors(self, side):
        return [self.facelet_matrix[0,col] for pos in Opticube.SOLVED_POS[side] 
                for col in range(self.facelet_matrix.shape[1]) 
                if sum (self.facelet_matrix[2:,col] == pos) == 3]
    
    def is_solved(self):
        return sum(sum(self.facelet_matrix[2:] == Opticube._facelet_matrix[2:])) == 162
        #return self.distance() == 1.0 # This check is also correct, but is ~3 times slower

    def state(self):
        return self.facelet_matrix[2:].T.reshape(162,)

    def nn_state(self):
        return self.facelet_matrix[2:, Opticube._movableindex].T.reshape(144,)
    
    def reset(self, state=None):
        if state is None: np.copyto(self.facelet_matrix, Opticube._facelet_matrix)
        else: self.facelet_matrix[2:,:] = state.reshape(54,3).T
        return self
    
    def distance(self):
        a_block = self.facelet_matrix[2:, Opticube._aindex]
        b_block = self.facelet_matrix[2:, Opticube._bindex]
        return sum(np.linalg.norm(b_block - a_block, axis=0)) / Opticube._DFACTOR
    
    def distance_simple(self):
        return sum(np.sum(self.facelet_matrix[2:,:] == Opticube._facelet_matrix[2:,:], axis=0) == 3)
    
    def scramble(self, sz=64):
        self.reset() # Always scrambles from a solved/reset state
        for action in np.random.randint(len(Opticube.MOVES), size=sz): 
            self.rotate(Opticube.MOVES[action])
        return self
    
    def trace_scramble(self, sz=20, apply_moves=False):
    #{
        moves = [Opticube.MOVES[action] for action in np.random.randint(len(Opticube.MOVES), size=sz)]

        if apply_moves: self.rotate_seq(moves)
        return moves, [self.inverse(mv) for mv in reversed(moves)]
    #}

    def rotate(self, move):
        cindex = self.get_facelet_indices(side=move[0])
        self.facelet_matrix[2:, cindex] = Opticube.rotation \
            (move).dot(self.facelet_matrix[2:, cindex])
        return self
    
    def rotate_seq(self, path):
        for move in path: self.rotate(move)
        return self
#}