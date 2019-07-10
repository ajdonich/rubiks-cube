import re
import numpy as np
from collections import defaultdict, OrderedDict
from Opticube import Opticube, color_letr

class FaceletSolver:
#{
    # Use static accessor below
    _position_to_colorindex = {}

    @staticmethod
    def pos_to_colind(flet_position_tuple):
    #{
        if not FaceletSolver._position_to_colorindex:
        #{
            # Maps home positions to home color-indexes
            for flet in Opticube._facelet_matrix.T:
                FaceletSolver._position_to_colorindex[tuple(flet[2:])] = tuple(flet[0:2])
        #}
    
        return FaceletSolver._position_to_colorindex[flet_position_tuple]
    #}

    @staticmethod
    def flet_str(flet):
    #{
        homeci = FaceletSolver.pos_to_colind(tuple(flet[2:]))
        return f"{color_letr(homeci[0])}{homeci[1]} at {color_letr(flet[0])}{flet[1]}"
    #}

    @staticmethod
    def path_str(path):
    #{
        path_str = ''
        for mv in path: path_str += f'{color_letr(mv[0])}({mv[1]})::'
        return re.sub(r'::$', '', path_str)
    #}
    
    def __init__(self):
        self.sequence_hash = None
        self.heuristic_hash = None
    
    # Just a helper fcn for create_sequence_hash below
    def _update_sequence_hash(self, sequence_hash, wc_hash, k, wildcard, *moves):
    #{
        # Logic for wildcarding
        if k not in sequence_hash:
            sequence_hash[k] = set()
            wc_hash[k] = len(moves)

        # Note: moves must be reversed and inverted
        if wildcard or (len(moves) == wc_hash[k]):
            sequence_hash[k].add(tuple([Opticube.inverse(mv) for mv in reversed(moves)]))
    #}

    # Returned sequence_hash contains as keys the 1152 possible facelet permutations 
    # (i.e. each possible position for each of the 48 moveable color/index facelets).
    # Each key maps to a set of rotations-sequences necessary to return that facelet to
    # its home postion (this turns out is at most 3 rotations for edges, 2 for corners).
    # If wildcard is False, sequence_hash contains only the shortest rotation-sequences,
    # but if True, it will contain all valid 1, 2, and 3-rotation-sequences per facelet.
    def create_sequence_hash(self, wildcard=False):
    #{
        cube = Opticube()
        sequence_hash = {}
        wc_hash = {}
        
        # Compute for every edge/corner facelet
        for i in Opticube._movableindex:
        #{
            # One move solutions
            for mv1 in Opticube.MOVES:
                cube.reset().rotate(mv1)
                self._update_sequence_hash(sequence_hash, wc_hash, tuple(cube.facelet_matrix[:,i]), wildcard, mv1)

            # Two move solutions
            for mv1 in Opticube.MOVES:
                for mv2 in Opticube.MOVES:
                    cube.reset().rotate(mv1).rotate(mv2)
                    self._update_sequence_hash(sequence_hash, wc_hash, tuple(cube.facelet_matrix[:,i]), wildcard, mv1, mv2)

            # Three move solutions
            for mv1 in Opticube.MOVES:
                for mv2 in Opticube.MOVES:
                    for mv3 in Opticube.MOVES:
                        cube.reset().rotate(mv1).rotate(mv2).rotate(mv3)
                        self._update_sequence_hash(sequence_hash, wc_hash, tuple(cube.facelet_matrix[:,i]), wildcard, mv1, mv2, mv3)
        #}

        self.sequence_hash = sequence_hash
        return sequence_hash
    #}

    # Similar to sequence_hash, but maps keys (facelet tuples) to a single 
    # value: 1, 2 or 3 designating the mimimum number of rotations get home 
    def create_heuristic_hash(self, sequence_hash):
    #{
        heuristic_hash = {}
        for flet_t in sequence_hash: heuristic_hash[flet_t] = np.amin([len(path) for path in sequence_hash[flet_t]])
                    
        self.heuristic_hash = heuristic_hash
        return heuristic_hash
    #}
    
    def flet_numb_moves(self, cube, sequence_hash):
    #{
        homelist = sum(cube.facelet_matrix[2:, Opticube._movableindex] == 
                       Opticube._facelet_matrix[2:, Opticube._movableindex]) == 3

        return [1 if home else len(sequence_hash[tuple(flet)][0]) for home, flet in 
                zip(homelist, cube.facelet_matrix[:, Opticube._movableindex].T)]
    #}

    # Note: search_depth is a lower bound, as this fcn will not cease
    # search in the middle of a block of best_moves of a given vote count   
    def best_next_moves(self, cube, prev_move=None, search_depth=20):
    #{
        # Note: dict value is irrelevant, just
        # need the ordered set of keys/moves
        best_next_move = OrderedDict()

        votes, depth = 0, 0
        best_move_seq = self.best_sequence(cube)
        for seq in sorted(best_move_seq.items(), reverse=True, key=lambda kv: kv[1]):
        #{
            if seq[1] > votes:
                if depth > search_depth: break
                votes = seq[1]
            elif seq[1] < votes: votes = 0
            best_next_move[seq[0][0]] = 1
            depth += 1
        #}

        # Prevent single-move-reversal local minima rut
        if prev_move in best_next_move: del best_next_move[prev_move]
        return list(best_next_move)
    #}

    def best_sequence(self, cube):
    #{
        best_sequence = defaultdict(lambda: 0)
        for flet in cube.facelet_matrix[:, Opticube._movableindex].T:
            for path in self.sequence_hash[tuple(flet)]: best_sequence[path] += 1

        return best_sequence
    #}

    def heuristic(self, cube):
    #{
        # This heuristic is sum(2 * heruistic_hash[facelet]), minus a 1-point bonus for every solved facelet
        nhomelist = sum(cube.facelet_matrix[2:, Opticube._movableindex] == Opticube._facelet_matrix[2:, Opticube._movableindex]) != 3
        numb_moves = sum([(self.heuristic_hash[tuple(flet)] * 2) for flet in cube.facelet_matrix[:, Opticube._movableindex[nhomelist]].T])
        return numb_moves + sum([-1 for flet in cube.facelet_matrix[:, Opticube._movableindex[np.invert(nhomelist)]].T])
    #}

    def heuristic_idx(self, cube, flet_idx):
    #{
        # Same scoring as heuristic above, but for subdividing into the speedcuber CFOP algorthim parts
        nhomelist = sum(cube.facelet_matrix[2:, flet_idx] == Opticube._facelet_matrix[2:, flet_idx]) != 3
        numb_moves = sum([(self.heuristic_hash[tuple(flet)] * 2) for flet in cube.facelet_matrix[:, flet_idx[nhomelist]].T])
        return numb_moves + sum([-1 for flet in cube.facelet_matrix[:, flet_idx[np.invert(nhomelist)]].T])
    #}

    def restricted_moves(self, cube, flet_idx):
    #{
        # Finds only moves that rotate one of the currently unsolved facelets in flet_idx or its home location 
        nhomelist = sum(cube.facelet_matrix[2:, flet_idx] == Opticube._facelet_matrix[2:, flet_idx]) != 3
        from_inner  = cube.facelet_matrix[2:, flet_idx[nhomelist]].T.dot(Opticube._facelet_matrix[2:, Opticube._centers])
        to_inner    = Opticube._facelet_matrix[2:, flet_idx[nhomelist]].T.dot(Opticube._facelet_matrix[2:, Opticube._centers])

        if type(sum(np.concatenate((from_inner, to_inner), axis=1) > 0)) == int:
            print(flet_idx)
            print(cube.facelet_matrix[:, flet_idx])
            print(Opticube._facelet_matrix[:, flet_idx])
            print(nhomelist)
            print(from_inner)
            print(to_inner)

        valid_sides = sum(sum(np.concatenate((from_inner, to_inner), axis=1) > 0).reshape((2,-1)) > 0) > 0
        return [move for move in Opticube.MOVES if valid_sides[move[0]-1]]
    #}

    # def heuristic(self, cube):
    # #{
    #     homelist = sum(cube.facelet_matrix[2:, Opticube._movableindex] == Opticube._facelet_matrix[2:, Opticube._movableindex]) != 3
    #     return sum([self.heuristic_hash[tuple(flet)] for flet in cube.facelet_matrix[:, Opticube._movableindex[homelist]].T])
    # #}

    # def rubistic(self, cube):
    # #{
    #     # cube_state_hash = {}
    #     # for flet in cube.facelet_matrix[:, Opticube._movableindex].T:
    #     # #{
    #     #     rolled_out = []
    #     #     for path in self.sequence_hash[tuple(flet)]:
    #     #         path_t = tuple(path)
    #     #         if path_t in cube_state_hash: rolled_out.append(cube_state_hash[path_t])
    #     #         else: rolled_out.append(self.heuristic(Opticube(cube).rotate_seq(path)))
    #     # #}

    #     return np.amin([np.amin([self.heuristic(Opticube(cube).rotate_seq(path)) for path in self.sequence_hash[tuple(flet)]]) 
    #                     for flet in cube.facelet_matrix[:, Opticube._movableindex].T])
    # #}

    def best_end_moves(self, cube, sequence_hash):
    #{
        best_end_move = {mv: 0 for mv in Opticube.MOVES}
        for flet in cube.facelet_matrix[:, Opticube._movableindex].T:
            for path in sequence_hash[tuple(flet)]: best_end_move[path[-1]] += 1

        return best_end_move
    #}

    def print_sequence_hash(self, sequence_hash, exclusive=False):
    #{
        index = 1
        for k,v in sorted(sequence_hash.items()):
        #{
            ci = FaceletSolver.pos_to_colind(k[2:])
            szmv1, szmv2, szmv3 = 0, 0, 0
            
            for path in v:
                if len(path) == 1: szmv1 += 1
                elif len(path) == 2: szmv2 += 1
                elif len(path) == 3: szmv3 += 1

            if not exclusive:
                print(f"  {index} : {color_letr(k[0])}{k[1]} at {color_letr(ci[0])}{ci[1]} : unique paths {szmv1} : {szmv2} : {szmv3} ")
            
            else:
            #{
                paths = None
                if szmv1 > 0: paths = [FaceletSolver.path_str(path) for path in v if len(path) == 1]
                elif szmv2 > 0: paths = [FaceletSolver.path_str(path) for path in v if len(path) == 2]
                elif szmv3 > 0: paths = [FaceletSolver.path_str(path) for path in v if len(path) == 3]
                print(f"  {index} : {color_letr(k[0])}{k[1]} at {color_letr(ci[0])}{ci[1]} : {len(paths)} shortest paths: {paths}")
            #}
            
            index += 1
        #}
    #}
    
    def print_shortest_paths(self, cube, sequence_hash):
    #{
        path_strings = []
        for i in Opticube._movableindex:
        #{
            min_paths = []
            flet = tuple(cube.facelet_matrix[:,i])
            for min_length in [1,2,3]:
            #{
                if min_paths: break
                for path in sequence_hash[flet]:
                    if len(path) == min_length: min_paths.append(FaceletSolver.path_str(path))
            #}
        
            path_strings.append(f"{FaceletSolver.flet_str(flet)} : {min_paths}")            

            # homeci = FaceletSolver.pos_to_colind[tuple(flet[2:])]
            # path_strings.append(f"{color_letr(homeci[0])}{homeci[1]} at {color_letr(flet[0])}{flet[1]} : {min_paths}")
        #}
        
        for ps in sorted(path_strings): print(ps)
    #}
    
    def sequence_vote(self, cube, sequence_hash, in_moves=3):
    #{
        vote_tally = {}

#         for i in Opticube._tlayerindex:
#         for i in [0,1,2,3,5,6,7,8]:
        
        # Tally for every edge/corner facelet
        for i in Opticube._movableindex:
        #{
            flet = tuple(cube.facelet_matrix[:,i])
            for path in sequence_hash[flet]:
                if len(path) <= in_moves:
                    if path not in vote_tally: vote_tally[path] = []
                    vote_tally[path].append(flet)
        #}
            
        return sorted(vote_tally.items(), reverse=True, key=lambda kv_t: len(kv_t[1]))
    #}
    
    def move_vote(self, cube, sequence_hash, ):
    #{
        vote_tally = {(sd, ang): [] for sd, ang in Opticube.MOVES}
        
        for i in Opticube._movableindex:
            flet = tuple(cube.facelet_matrix[:,i])
            for path in sequence_hash[flet]: vote_tally[path[0]].append(flet)
            
        return sorted(vote_tally.items(), reverse=True, key=lambda kv_t: len(kv_t[1]))
    #}
    
    def print_solution_table(self, cube, sequence_hash):
    #{
        MAX_UNIQUE = 12
    
        state = cube.state()
        table = np.full((len(Opticube._movableindex), GODS_NUMBER), -1, dtype=int)       
        top_move, voters = self.move_vote(cube, sequence_hash)[0]
        
        # Initialize table
        for n in range(3):
            for i, ti in zip(Opticube._movableindex, range(table.shape[0])):
                for path in sequence_hash[tuple(cube.facelet_matrix[:,i])]:
                    if n < len(path): table[ti, n] = Opticube.ACTIONS[path[n]]
                    else: break
        
        print(table)
        
#         for n in range(GODS_NUMBER):
#             for i in Opticube._movableindex:
            
#                 for path in sequence_hash[tuple(cube.facelet_matrix[:,i])]:
#                     if path[0] == top_move: table[i,n] = Opticube.ACTIONS[top_move]
    #}
    
    # Plays max 20 moves to solve the cube
    def solve_cube(self, cube, sequence_hash, move_sequence=None, nmoves=20):
    #{
        if move_sequence is None: move_sequence = []
        if cube.is_solved(): print(f"Solved cube in {GODS_NUMBER - nmoves} moves.")
        elif nmoves == 0:    print("GAME OVER: moves expired") 
        else:
        #{
            votes = self.sequence_vote(cube, sequence_hash, in_moves=nmoves)
            for move in votes[0][0]: cube.rotate(move); move_sequence.append(move)
            self.solve_cube(cube, sequence_hash, move_sequence, nmoves=(nmoves-len(votes[0][0])))
        #}
        
        return move_sequence
    #}

    def solve_cube_shunt(self, cube, sequence_hash, move_sequence=None, nmoves=20):
    #{
        votes = [((), self.sequence_vote(cube, sequence_hash, in_moves=nmoves))]
        
        # Shunt moves search
        for mv1 in Opticube.MOVES:
            cube.rotate[mv1]
            votes.append(((mv1,), self.sequence_vote(cube, sequence_hash, in_moves=nmoves)))
            for mv2 in Opticube.MOVES:
                votes.append(((mv1,mv2), self.sequence_vote(cube, sequence_hash, in_moves=nmoves)))
                for mv3 in Opticube.MOVES:
                    votes.append(((mv1,mv2,mv3), self.sequence_vote(cube, sequence_hash, in_moves=nmoves)))
                    
        
        for move in votes[0][0]: cube.rotate_s(move); move_sequence.append(move)
        self.solve_cube(cube, sequence_hash, move_sequence, nmoves=(nmoves-len(votes[0][0])))
      
        return move_sequence
    #}
#}