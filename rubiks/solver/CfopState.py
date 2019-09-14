import time
import numpy as np

from rubiks.model.CfopCube import CfopCube, CROSS, F2L, OLL, PLL, FULL
from rubiks.model.VectorCube import VectorCube, WHITE_CB, ORANGE_CB, GREEN_CB, RED_CB, BLUE_CB, YELLOW_CB

class CfopState:
#{
    cfop_queue = [CROSS, F2L, OLL, PLL, 'Done']
    INVERSE_COLORS = [None, YELLOW_CB, RED_CB, BLUE_CB, ORANGE_CB, GREEN_CB, WHITE_CB]
    
    def __init__(self, fsolver, state):
        self.fsolver = fsolver
        self.state = state
    
    def increment_state(self):
        idx = next(i for i, state in enumerate(CfopState.cfop_queue) if self.state == state)
        self.state = CfopState.cfop_queue[idx+1]
        return self.state
    
    def heuristic(self, cfop_cube):
        return self.fsolver.heuristic_idx(cfop_cube, self.flet_index())
    
    def expand_moves(self, cfop_cube):
        return self.fsolver.restricted_moves(cfop_cube, self.flet_index())
    
    # Interface methods to override in child classes
    def is_complete(self, cfop_cube, init_color=WHITE_CB): pass
    def flet_index(self, init_color=WHITE_CB): pass
#}

class FullCubeState(CfopState):
#{
    def __init__(self, fsolver):
        super().__init__(fsolver, FULL)
    
    def flet_index(self, init_color=WHITE_CB):
        return CfopCube._minimalindex
    
    def heuristic(self, cfop_cube):
        return self.fsolver.order_heuristic(cfop_cube)
    
    def is_complete(self, cfop_cube, init_color=WHITE_CB):
        return cfop_cube.solved(flet_index=self.flet_index())
#}

class CrossState(CfopState):
#{
    def __init__(self, fsolver):
        super().__init__(fsolver, CROSS)
    
    def flet_index(self, init_color=WHITE_CB):
        return CfopCube.CFOP_IDXS[CROSS, init_color]
    
    def is_complete(self, cfop_cube, init_color=WHITE_CB):
        return cfop_cube.solved(flet_index=self.flet_index())
#}

class F2LStateOrbit(CfopState):
#{
    substate_queue = ['SINGLE_PASS', 'WITHIN_5ORBIT', 'F2L_CORNER', 'Done']

    # Access through static methods below
    _orbit_pairs = {}

    @staticmethod
    def get_orbit_pairs(f2l_pair):
    #{
        if not F2LStateOrbit._orbit_pairs:
        #{
            start = time.time()
            print("Calculating 5-deep states (requires approx 2-3 mins to complete)...")
            dstates = F2LStateOrbit._five_deep()
            print(f"Calculation completed in {int(time.time() - start)} sec")

            posdict = { k[1]: set() for k in dstates.keys() }
            for k in dstates.keys():
                for state in dstates[k]: 
                    posdict[k[1]].add(tuple(VectorCube().reset(state=state).
                                            facelet_matrix[2:, k[1]].T.flatten()))

            F2LStateOrbit._orbit_pairs = { k: np.zeros((len(posset),3,2),
                dtype=int) for k, posset in posdict.items() }
            
            for k, posset in posdict.items():
                for i, pos_pair in enumerate(posset):
                    F2LStateOrbit._orbit_pairs[k][i,:,:] = \
                        np.array(pos_pair).reshape((2,3)).T
        #}

        return F2LStateOrbit._orbit_pairs[tuple(f2l_pair)]
    #}
    
    @staticmethod
    def _five_deep(cube=None, dstates=None, depth=0):
    #{
        if cube is None: cube = VectorCube()
        if dstates is None: dstates = {tuple((depth, tuple(cn_pair))): [] for cn_pair in 
            CfopCube.CFOP_IDXS[(F2L, WHITE_CB)][:,:2] for depth in range(5)}

        if depth < 5:
        #{
            for mv in VectorCube.MOVES:
                mv_cube = VectorCube(cube).rotate(mv)
                F2LStateOrbit._five_deep(mv_cube, dstates, depth+1)
                index = CfopCube.CFOP_IDXS[(CROSS, WHITE_CB)]
                for cn_pair in CfopCube.CFOP_IDXS[(F2L, WHITE_CB)][:,:2]:
                    key = tuple((depth, tuple(cn_pair)))
                    if (mv_cube.solved(flet_index=index)): dstates[key].append(mv_cube.state()) #; print(mv, index)
                    index = np.concatenate((index, cn_pair))
        #}
        
        return dstates
    #}

    def __init__(self, fsolver):
        super().__init__(fsolver, F2L)
        self.substate = None
        self.f2l_pair = None
        self.f2l_idx = []
    
    def set_f2l_pair(self, idx_pair):
        self.f2l_pair = idx_pair
        self.f2l_idx.extend(idx_pair)
        self.substate = 'SINGLE_PASS'
    
    def increment_substate(self):
        if self.substate == 'Done': return self.substate
        idx = next(i for i, substate in enumerate(F2LStateOrbit.substate_queue) if self.substate == substate)
        self.substate = F2LStateOrbit.substate_queue[idx+1]
        return self.substate
    
    # f2l_idx is accumulated f2l_pairs 
    def flet_index(self, init_color=WHITE_CB):
        return np.concatenate((CfopCube.CFOP_IDXS[(CROSS, init_color)],
                               np.array(self.f2l_idx, dtype=int)))
    
    def get_unsolved_pairs(self, cube, init_color=WHITE_CB):
    #{
        self.f2l_idx, unsolved_pairs = [], []
        for ci_pair in CfopCube.CFOP_IDXS[(F2L, init_color)][:,:2]:
            if cube.solved(flet_index=ci_pair): self.f2l_idx.extend(ci_pair)
            else: unsolved_pairs.append(ci_pair)
        
        return unsolved_pairs
    #}
    
    def is_complete(self, cfop_cube, init_color=WHITE_CB):
    #{
        # Always check for full step completion first
        if cfop_cube.solved(flet_index=self.flet_index()):
            self.substate = 'Done'
            return True

        if self.substate == 'WITHIN_5ORBIT':
            orbit_pairs = F2LStateOrbit.get_orbit_pairs(self.f2l_pair)
            pair_block = np.broadcast_to(cfop_cube.facelet_matrix[2:, self.f2l_pair], orbit_pairs.shape)
            return (cfop_cube.solved(flet_index=self.flet_index()[:-2]) and
                    (sum(np.sum(orbit_pairs == pair_block, axis=(1,2)) == 6) > 0))
        
        return False
    #}
#}

class OLLState(CfopState):
#{
    def __init__(self, fsolver):
        super().__init__(fsolver, OLL)
        self.human_algs = self.load_algorithms()
        
    def load_algorithms(self):
        with open('../data/CFOP_OLL_ALGS.TXT') as f_handle:
            return [line.rstrip('\n').replace(" ", "").split(',')
                    for line in f_handle if line != '\n']
    
    def flet_index(self, init_color=WHITE_CB):
        return CfopCube.CFOP_IDXS[OLL, init_color]
    
    def is_complete(self, cfop_cube, init_color=WHITE_CB):
    #{  
        cent_idx = CfopCube._centers[CfopState.INVERSE_COLORS[init_color]-1]
        f2l_idx = np.concatenate((CfopCube.CFOP_IDXS[(CROSS, init_color)], 
                                  CfopCube.CFOP_IDXS[(F2L, init_color)].flatten()))
        
        return (cfop_cube.solved(flet_index=f2l_idx) and
                sum(cfop_cube.facelet_matrix[2:, self.flet_index()].T.dot
                    (CfopCube._facelet_matrix[2:, cent_idx]) > 6) == 8)
    #}
#}

class PLLState(CfopState):
#{
    def __init__(self, fsolver):
        super().__init__(fsolver, PLL)
        self.human_algs = self.load_algorithms()
    
    def load_algorithms(self):
        with open('../data/CFOP_PLL_ALGS.TXT') as f_handle:
            return [line.rstrip('\n').replace(" ", "").split(',')
                    for line in f_handle if line != '\n']
    
    def flet_index(self, init_color=WHITE_CB):
        return CfopCube._movableindex
    
    def is_complete(self, cfop_cube, init_color=WHITE_CB):
        return cfop_cube.solved()
#}