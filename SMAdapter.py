import math, re
import numpy as np
from Observer import Observable
from VectorCube import VectorCube

class SMAdapter(Observable):
#{
    # Rubik's Cube Singmaster Notation:
    # Clockwise face rotations:   [U, L, F, R, B, D]
    # Counterclockwise face:      [U', L', F', R', B', D']
    # Slice turns:                [M, M', E, E', S, S']
    # Double layer turns:         [u, l, f, r, b, d]
    # Inverse double layer turns: [u', l', f', r', b', d']
    # Whole cube rotations:       [X, X', Y, Y', Z, Z']

    # Full list of valid singmaster moves
    SM_NAMES = ["U", "L", "F", "R", "B", "D", 
                "U\'", "L\'", "F\'", "R'", "B\'", "D'", 
                "X", "Y", "Z", "X\'", "Y\'", "Z\'", 
                "M", "E", "S", "M\'", "E\'", "S\'", 
                "u", "l", "f", "r", "b", "d", 
                "u\'", "l\'", "f\'", "r'", "b\'", "d'",
                "U2", "L2", "F2", "R2", "B2", "D2", 
                "X2", "Y2", "Z2", "M2", "E2", "S2", 
                "u2", "l2", "f2", "r2", "b2", "d2"]

    # Global centers/bases
    Xi, Yi, Zi    = (3,0,0),  (0,3,0),  (0,0,3)
    nXi, nYi, nZi = (-3,0,0), (0,-3,0), (0,0,-3)
    GLOBAL_BASES = [Xi, Yi, Zi, nXi, nYi, nZi]

    _global_indicies = {
        Zi:  np.array([0,1,2,3,4,5,6,7,8,9,10,11,18,19,20,27,28,29,36,37,38]),
        nYi: np.array([0,3,6,9,10,11,12,13,14,15,16,17,18,21,24,38,41,44,45,48,51]),
        Xi:  np.array([6,7,8,11,14,17,18,19,20,21,22,23,24,25,26,27,30,33,45,46,47]),
        Yi:  np.array([2,5,8,20,23,26,27,28,29,30,31,32,33,34,35,36,39,42,47,50,53]),
        nXi: np.array([0,1,2,9,12,15,29,32,35,36,37,38,39,40,41,42,43,44,51,52,53]),
        nZi: np.array([15,16,17,24,25,26,33,34,35,42,43,44,45,46,47,48,49,50,51,52,53]),
        (Xi, 'slice'): np.array([3,4,5,10,13,16,28,31,34,48,49,50]),
        (Yi, 'slice'): np.array([1,4,7,19,22,25,37,40,43,46,49,52]),
        (Zi, 'slice'): np.array([12,13,14,21,22,23,30,31,32,39,40,41])
    }

    _divider_indicies = { Xi: np.array([2]), nXi: np.array([4]),
                          Yi: np.array([3]), nYi: np.array([1]),
                          Zi: np.array([0]), nZi: np.array([5]),
                          (Xi, 'slice'): np.array([8,10]), 
                          (Yi, 'slice'): np.array([7,9]),
                          (Zi, 'slice'): np.array([6,11])
    }
    
    _face_rotations = {                                       # Face color (in init orientation):
        "U": (Zi, -90),  "U\'": (Zi, 90),   "U2": (Zi, 180),  # WHITE_CB    i.e. ( 0,  0,  3)
        "L": (nYi, 90),  "L\'": (nYi, -90), "L2": (nYi, 180), # ORANGE_CB   i.e. ( 0, -3,  0)
        "F": (Xi, -90),  "F\'": (Xi, 90),   "F2": (Xi, 180),  # GREEN_CB    i.e. ( 3,  0,  0)
        "R": (Yi, -90),  "R\'": (Yi, 90),   "R2": (Yi, 180),  # RED_CB      i.e. ( 0,  3,  0)
        "B": (nXi, 90),  "B\'": (nXi, -90), "B2": (nXi, 180), # BLUE_CB     i.e. (-3,  0,  0)
        "D": (nZi, 90),  "D\'": (nZi, -90), "D2": (nZi, 180)  # YELLOW_CB   i.e. ( 0,  0, -3)
    }

    _cube_rotations = {
        "X": (Yi, -90), "X\'": (Yi, 90), "X2": (Yi, 180), 
        "Y": (Zi, -90), "Y\'": (Zi, 90), "Y2": (Zi, 180), 
        "Z": (Xi, -90), "Z\'": (Xi, 90), "Z2": (Xi, 180)
    }
    
    _slice_turns = {
        "M": ((Yi, 90), "R", "L\'"),  "M\'": ((Yi, -90), "R\'", "L"), "M2": ((Yi, 180), "R2", "L2"),
        "E": ((Zi, 90), "U", "D\'"),  "E\'": ((Zi, -90), "U\'", "D"), "E2": ((Zi, 180), "U2", "D2"),
        "S": ((Xi, -90), "B", "F\'"), "S\'": ((Xi, 90),  "B\'", "F"), "S2": ((Xi, 180), "B2", "F2")
    }

    _double_layer_turns = {
        "u": ((Zi, -90), "U", "D"), "u\'": ((Zi, 90), "U\'", "D\'"),  "u2": ((Zi, 180), "U2", "D2"),
        "l": ((Yi, 90), "L", "R"),  "l\'": ((Yi, -90), "L\'", "R\'"), "l2": ((Yi, 180), "L2", "R2"),
        "f": ((Xi, -90), "F", "B"), "f\'": ((Xi, 90), "F\'", "B\'"),  "f2": ((Xi, 180), "F2", "B2"),
        "r": ((Yi, -90), "R", "L"), "r\'": ((Yi, 90), "R\'", "L\'"),  "r2": ((Yi, 180), "R2", "L2"),
        "b": ((Xi, 90), "B", "F"),  "b\'": ((Xi, -90), "B\'", "F\'"), "b2": ((Xi, 180), "B2", "F2"),
        "d": ((Zi, 90), "D", "U"),  "d\'": ((Zi, -90), "D\'", "U\'"), "d2": ((Zi, 180), "D2", "U2")
    }

    # Access through static method below
    _basis_rotation_matrices = {}
    
    @staticmethod
    def basis_rotation(gbase_angle_tuple):
    #{
        if not SMAdapter._basis_rotation_matrices:
        #{
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                SMAdapter._basis_rotation_matrices[(SMAdapter.Xi, deg)]  = Rx
                SMAdapter._basis_rotation_matrices[(SMAdapter.Yi, deg)]  = Ry
                SMAdapter._basis_rotation_matrices[(SMAdapter.Zi, deg)]  = Rz
                SMAdapter._basis_rotation_matrices[(SMAdapter.nXi, deg)] = Rx
                SMAdapter._basis_rotation_matrices[(SMAdapter.nYi, deg)] = Ry
                SMAdapter._basis_rotation_matrices[(SMAdapter.nZi, deg)] = Rz
            #}
        #}

        return SMAdapter._basis_rotation_matrices[gbase_angle_tuple]
    #}
    
    # Helpers for generating inbetween animation frames
    def _getRx(c,s): return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def _getRy(c,s): return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def _getRz(c,s): return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    _radians = { 90: math.pi/2, -90: -math.pi/2, 180: math.pi }
    _gen_rotations = { Xi: _getRx, Yi: _getRy, Zi: _getRz,
                       nXi: _getRx, nYi: _getRy, nZi: _getRz }

    @staticmethod
    def inbetween_rotations(sm_move, steps=5, fast_half_turns=False):
        if (not fast_half_turns) and (sm_move[1] == 180): steps *= 2
        rmats = [SMAdapter._gen_rotations[sm_move[0]](math.cos(rad), math.sin(rad))
                 for rad in np.linspace(0, SMAdapter._radians[sm_move[1]], num=steps)]
        rmats[0], rmats[steps-1] = np.round(rmats[0]).astype(int), np.round(rmats[steps-1]).astype(int)
        return rmats

    def __init__(self, local_cube, local_x=[1,0,0], local_y=[0,1,0], local_z=[0,0,1]):
    #{
        # Constructs queue for CubeView observers
        super(SMAdapter, self).__init__()

        # Cube keeps its own local facelet representations, while SMAdapter
        # just keeps local_basis vectors for whole cube's position in global space
        self.local_cube = local_cube
    
        # Note: transpose superfluous with default-bases but needed in general 
        self.local_basis = np.array([local_x, local_y, local_z]).T
        self.inv_basis = np.array(np.linalg.inv(self.local_basis), dtype=int)
        
        # Slice and double layer turns are all some combo
        # of the basic face and whole-cube rotations below
    #}
    
    def get_global_positions(self):
        return np.matmul(self.local_basis, self.local_cube.facelet_matrix[2:])

    # Help fcn for rotate_singmaster
    def apply_face_rotation(self, sm_move, sm_paired_move=None):
    #{
        # A global rotation in local coordinates (as defined by local_basis 
        # vectors as columns of matrix A) is: A^(-1) @ Global_Rotation @ A
        rmat = SMAdapter.basis_rotation(sm_move)
        local_rmat = self.inv_basis @ rmat @ self.local_basis
        
        # Any local vector in local frame is described in global frame as: A @ local_vector
        gcenters = np.matmul(self.local_basis, VectorCube._facelet_matrix[2:, VectorCube._centers])
        local_side = np.nonzero(sum(gcenters == np.broadcast_to(sm_move[0], (6, 3)).T) == 3)[0][0] + 1

        # Then execute the local rotation to the local_cube's frame 
        cindex = self.local_cube.get_facelet_indices(side=local_side)
        self.local_cube.facelet_matrix[2:, cindex] = np.matmul \
            (local_rmat, self.local_cube.facelet_matrix[2:, cindex])

        # Rotate paired face for slice turns
        if sm_paired_move is not None:
            local_side = np.nonzero(sum(gcenters == np.broadcast_to(sm_paired_move[0], (6, 3)).T) == 3)[0][0] + 1
            cindex = self.local_cube.get_facelet_indices(side=local_side)
            self.local_cube.facelet_matrix[2:, cindex] = np.matmul \
                (local_rmat, self.local_cube.facelet_matrix[2:, cindex])
    #}

    # Help fcn for rotate_singmaster
    def apply_basis_rotation(self, sm_move):
    #{
        # Whole cube/basis rotations in global frame
        rmat = SMAdapter.basis_rotation(sm_move)
        self.local_basis = np.matmul(rmat, self.local_basis)
        self.inv_basis = np.array(np.linalg.inv(self.local_basis), dtype=int)
    #}

    # In general, FIRST notify CubeView observers w/resepct to global frame,
    # THEN apply corresponding local face rotation(s), THEN basis rotation
    def rotate_singmaster(self, sm_name):
    #{
        if sm_name in SMAdapter._face_rotations:
            sm_move = SMAdapter._face_rotations[sm_name]
            gindex = SMAdapter._global_indicies[sm_move[0]]
            dindex = SMAdapter._divider_indicies[sm_move[0]]
            self.notify_observers(SMAdapter.inbetween_rotations(sm_move), mindex=gindex, dividx=dindex)
            self.apply_face_rotation(sm_move)

        elif sm_name in SMAdapter._cube_rotations:
            sm_move = SMAdapter._cube_rotations[sm_name]
            gindex, dindex = np.arange(0,54), np.arange(0,12)
            self.notify_observers(SMAdapter.inbetween_rotations(sm_move), mindex=gindex, dividx=dindex)
            self.apply_basis_rotation(sm_move)

        elif sm_name in SMAdapter._slice_turns:
            sm_move, sm_fname1, sm_fname2 = SMAdapter._slice_turns[sm_name]
            gindex = SMAdapter._global_indicies[(sm_move[0], 'slice')] 
            dindex = SMAdapter._divider_indicies[(sm_move[0], 'slice')] 
            self.notify_observers(SMAdapter.inbetween_rotations(sm_move), mindex=gindex, dividx=dindex)
            self.apply_face_rotation(SMAdapter._face_rotations[sm_fname1], SMAdapter._face_rotations[sm_fname2])
            self.apply_basis_rotation(sm_move)

        elif sm_name in SMAdapter._double_layer_turns:
            sm_move, sm_name1, sm_fname = SMAdapter._double_layer_turns[sm_name]
            gindex = np.concatenate((SMAdapter._global_indicies[(sm_move[0], 'slice')],
                                     SMAdapter._global_indicies[SMAdapter._face_rotations[sm_name1][0]]))
            dindex = np.concatenate((SMAdapter._divider_indicies[(sm_move[0], 'slice')],
                                     SMAdapter._divider_indicies[SMAdapter._face_rotations[sm_name1][0]]))
            self.notify_observers(SMAdapter.inbetween_rotations(sm_move), mindex=gindex, dividx=dindex)
            self.apply_face_rotation(SMAdapter._face_rotations[sm_fname])
            self.apply_basis_rotation(sm_move)
        
        return self
    #}
    
    def inverse(self, sm_name):
    #{
        result = re.match(r"([ULFRBDXYZMESulfrbd])\'", sm_name)
        if result: return result.group(1)

        result = re.match(r"[ULFRBDXYZMESulfrbd]2", sm_name)
        if result: return (result.group(0))

        result = re.match(r"[ULFRBDXYZMESulfrbd]", sm_name)
        if result: return (result.group(0) + "'")

        return None
    #}
    
    def trace_scramble(self, sz=20):
    #{
        # Assures non-redundant sequence of moves (i.e. no-back-to-back-rotations-of-same-side) 
        cands = [SMAdapter.SM_NAMES[index] for index in np.random.randint(len(SMAdapter.SM_NAMES), size=(sz*2))]
        sm_moves = [cands[i] for i in range(len(cands)) if (i == 0) or (cands[i-1][0] != cands[i][0])]
        
        return sm_moves[:sz], [self.inverse(mv) for mv in reversed(sm_moves[:sz])]
    #}

#}