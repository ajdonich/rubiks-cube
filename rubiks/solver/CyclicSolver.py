import time
import numpy as np
import matplotlib.pyplot as plt

from rubiks.model.DirectCube import DirectCube
from rubiks.model.VectorCube import VectorCube, SIDES, WHITE_CB, ORANGE_CB, GREEN_CB, RED_CB, BLUE_CB, YELLOW_CB

class CyclicSolver:
#{
    def __init__(self):
        self.cycles = self.gen_rhythmic_cycles()
        self.periods = self.gen_rhythmic_periods()
    
    def cycle_solve(self, starting_cube, heuristic_fcn, nmoves='length', verbose=False):
    #{
        # These for algorithm metrics
        rot_time, h_time, i_time = [],[],[]
        start = time.time()

        cube = DirectCube(starting_cube)
        best_state = cube.state()
        is_new_best_state = True
        all_moves = []
        best_moves = []
        best_value = 0 

        rollout_cache = {}
        while is_new_best_state:
        #{
            iteration_a = time.time()
            is_new_best_state = False
            for moves, period in zip(self.cycles, self.periods):
            #{
                rollout_dist = nmoves
                if nmoves == 'length':   rollout_dist = len(moves)
                elif nmoves == 'period': rollout_dist = period
                elif nmoves > period:    rollout_dist = period
                
                applied_moves = []
                rollout_cube = DirectCube(cube)    
                for j in range(rollout_dist):
                #{
                    # Rotations
                    a = time.time()
                    mv = moves[j%len(moves)]
                    applied_moves.append(mv)
                    rollout_cube.rotate(mv)
                    rot_time.append(time.time() - a)

                    # Heuristics
                    b = time.time()
                    k = tuple(rollout_cube.state())
                    if k not in rollout_cache:
                        rollout_cache[k] = heuristic_fcn(rollout_cube)
                        if rollout_cache[k] > best_value:
                            best_value = rollout_cache[k]
                            best_moves = list(applied_moves)
                            best_state = rollout_cube.state()
                            is_new_best_state = True
                    h_time.append(time.time() - b)
                #}
            #}
            
            # Store if new best cube rolled out
            if is_new_best_state:
                cube.reset(state=best_state)
                all_moves.extend(best_moves)
            
            i_time.append(time.time() - iteration_a)
        #}
        
        if verbose: 
        #{
            total = time.time() - start
            print(f"Cycle rollout distance: {nmoves}")
            print(f"Number of iterations: {len(i_time)}")
            print("Total run time:", total, "sec")
            print(f"Time rotating cube: {sum(rot_time)} sec (" + "{:.2f}".format(100*sum(rot_time)/total) + "%)")
            print(f"Time calcing heuristics: {sum(h_time)} sec (" + "{:.2f}".format(100*sum(h_time)/total) + "%)")
            print(f"Average time per iteration: {np.average(i_time)} sec")
            print(f"Rollout cache size: {len(rollout_cache)}\n")

            # plt.figure(1)
            # plt.plot(i_time)
            # plt.title(f'Time Per Iteration')
            # plt.ylabel('Seconds')
            # plt.xlabel('Iteration')
        #}
        
        return cube, all_moves, i_time
    #}

    def generate_perturbations(self, cube, moves):
    #{
        perturbations = []
        reference_cube = DirectCube()
        reverse_cube = DirectCube(cube)
        
        ishomelist = sum(reverse_cube.facelet_matrix[2:] == VectorCube._facelet_matrix[2:]) == 3
        homeidx, nhomeidx = np.nonzero(ishomelist)[0], np.nonzero(np.logical_not(ishomelist))[0]
        home_compv = sum(sum(reference_cube.compare(reverse_cube, flet_index=homeidx)) == 3)
        nhome_compv = sum(sum(reference_cube.compare(reverse_cube, flet_index=nhomeidx)) == 3)
        
        for i, inv_mv in reversed([(j, VectorCube.inverse(mv)) for j, mv in enumerate(moves)]):
        #{
            reverse_cube.rotate(inv_mv)
            reference_cube.rotate(inv_mv)
            for mv in VectorCube.MOVES:
            #{
                if VectorCube.inverse(mv) != inv_mv:
                    pert_cube = DirectCube(reverse_cube).rotate(mv)
                    hcomp = sum(sum(reference_cube.compare(pert_cube, flet_index=homeidx)) == 3)
                    nhcomp = sum(sum(reference_cube.compare(pert_cube, flet_index=nhomeidx)) == 3)
                    if (nhcomp > nhome_compv): perturbations.append(tuple((i, mv, DirectCube(pert_cube))))
            #}
        #}
        
        return sorted(perturbations)
    #}

    def gen_rhythmic_cycles(self):
    #{
        # The set of single-sided full-period rotation sequences 
        rhythms = [(90,-90), (-90,90), (180,180),
                   (180,90,90), (90,180,90), (90,90,180), 
                   (180,-90,-90), (-90,180,-90), (-90,-90,180),
                   (90,90,90,90), (-90,-90,-90,-90)]

        # The set of two-side/two-axis combinations
        side_combos = [(WHITE_CB, ORANGE_CB),(WHITE_CB, GREEN_CB),(WHITE_CB, RED_CB),(WHITE_CB, BLUE_CB),
                       (ORANGE_CB, GREEN_CB),(ORANGE_CB, BLUE_CB),(ORANGE_CB, YELLOW_CB),
                       (GREEN_CB, RED_CB),(GREEN_CB, YELLOW_CB),
                       (RED_CB, BLUE_CB),(RED_CB,YELLOW_CB),
                       (BLUE_CB, YELLOW_CB)]
        
        self.cycles = []
        for sd1, sd2 in side_combos:
            for r1 in rhythms:
                for r2 in rhythms:
                    if (len(r1) == len(r2)): self.cycles.extend(self._combine_sym(sd1, sd2, r1, r2))                        
                    elif (abs(len(r1) - len(r2)) == 1): self.cycles.append(self._combine_asym(sd1, sd2, r1, r2))
        
        return self.cycles
    #}

    # def gen_tri_rhythmic_cycles(self):
    # #{
    #     # The set of single-sided full-period rotation sequences 
    #     rhythms = [(90,-90), (-90,90), (180,180),
    #                (180,90,90), (90,180,90), (90,90,180), 
    #                (180,-90,-90), (-90,180,-90), (-90,-90,180),
    #                (90,90,90,90), (-90,-90,-90,-90)]

    #     # The set of three-side combinations (translates to 120 of them)
    #     tri_sides = sorted({(sd1,sd2,sd3) for sd1 in SIDES for sd2 in SIDES for sd3 in SIDES 
    #                        if (sd1 != sd2) and (sd1 != sd3) and (sd2 != sd3)})
        
    #     self.cycles = []
    #     for sd1, sd2, sd3 in tri_sides:
    #         for r1 in rhythms:
    #             for r2 in rhythms:
    #                 for r3 in rhythms:
    #                     if (len(r1) == len(r2)): self.cycles.extend(self._combine_sym(sd1, sd2, r1, r2))                        
    #                     elif (abs(len(r1) - len(r2)) == 1): self.cycles.append(self._combine_asym(sd1, sd2, r1, r2))
        
    #     return self.cycles
    # #}

    def gen_rhythmic_periods(self):
    #{
        self.periods = []
        for cymoves in self.cycles:
        #{
            cube = VectorCube()
            self.periods.append(0)

            solved = False
            while not solved:
            #{
                for mv in cymoves:
                    solved = cube.rotate(mv).is_solved()
                    self.periods[-1] += 1
                    if solved: break
            #}
        #}

        return self.periods
    #}

    # Helper for generate_move_cycles
    def _combine_asym(self, sd1, sd2, r1, r2):
    #{
        if (len(r1) == 3) and (len(r2) == 2): 
            return [(sd1, r1[0]), (sd2, r2[0]), (sd1, r1[1]), (sd2, r2[1]), (sd1, r1[2])]
        elif (len(r1) == 2) and (len(r2) == 3): 
            return [(sd2, r2[0]), (sd1, r1[0]), (sd2, r2[1]), (sd1, r1[1]), (sd2, r2[2])]
        elif (len(r1) == 4) and (len(r2) == 3): 
            return [(sd1, r1[0]), (sd2, r2[0]), (sd1, r1[1]), (sd2, r2[1]), (sd1, r1[2]), (sd2, r2[2]), (sd1, r1[3])]
        elif (len(r1) == 3) and (len(r2) == 4): 
            return [(sd2, r2[0]), (sd1, r1[0]), (sd2, r2[1]), (sd1, r1[1]), (sd2, r2[2]), (sd1, r1[2]), (sd2, r2[3])]
        
        return None
    #}
    
    # Helper for generate_move_cycles
    def _combine_sym(self, sd1, sd2, r1, r2):
    #{
        if (len(r1) == 2) and (len(r2) == 2):
            return ([(sd1, r1[0]), (sd2, r2[0]), (sd1, r1[1]), (sd2, r2[1])], 
                    [(sd2, r2[0]), (sd1, r1[0]), (sd2, r2[1]), (sd1, r1[1])])
        elif (len(r1) == 3) and (len(r2) == 3): 
            return ([(sd1, r1[0]), (sd2, r2[0]), (sd1, r1[1]), (sd2, r2[1]), (sd1, r1[2]), (sd2, r2[2])], 
                    [(sd2, r2[0]), (sd1, r1[0]), (sd2, r2[1]), (sd1, r1[1]), (sd2, r2[2]), (sd1, r1[2])])
        elif (len(r1) == 4) and (len(r2) == 4):
            return ([(sd1, r1[0]), (sd2, r2[0]), (sd1, r1[1]), (sd2, r2[1]), (sd1, r1[2]), (sd2, r2[2]), (sd1, r1[3]), (sd2, r2[3])], 
                    [(sd2, r2[0]), (sd1, r1[0]), (sd2, r2[1]), (sd1, r1[1]), (sd2, r2[2]), (sd1, r1[2]), (sd2, r2[3]), (sd1, r1[3])])
        
        return None, None
    #}
#}