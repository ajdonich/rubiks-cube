import numpy as np
import random, re, time
from IPython.display import clear_output

from rubiks.model.CubeView import CubeView
from rubiks.model.DirectCube import DirectCube
from rubiks.model.VectorCube import VectorCube, color_name, color_letr

from rubiks.solver.UCTNode import HStateNode, HeuristicState


class DirectSolver:
#{
    ref_moves = None
    prnt_time = time.time()

    @staticmethod
    def output_state_progress(solved, direct_cube, moves):
    #{
        if not moves: print(f"Solved direct cube: {solved} (zero moves applied)\n")
        else:
        #{
            print(f"Solved direct cube: {solved}")
            print(f"  Numb tree nodes: {HStateNode.total_nodes}, node visits: {HStateNode.total_visits}")
            print(f"  Numb moves to best cube: {len(moves)} (best scaled node value: {HStateNode.top_value})")
            print(f"  Moves: {[f'({color_letr(mv[0])}:{mv[1]})' for mv in moves]}\n")
            HStateNode.reset_tree()
            
            view = CubeView(direct_cube)
            for depth, mv in enumerate(moves):
            #{
                direct_cube.rotate(mv)
                if((len(moves) - depth+1) < 20):
                    # value = hterse(DirectSolver.generate_hstate(direct_cube))
                    # view.push_snapshot(caption=f"{color_name(mv[0])} ({mv[1]}) : {value}")
                    view.push_snapshot(caption=f"{color_name(mv[0])} ({mv[1]}) : {DirectSolver.generate_hstate(direct_cube)}")
            #}

            view.draw_snapshots()
        #}
        
        return direct_cube
    #}
    
    @staticmethod
    def output_running_status(node_path_list):
    #{
        if (time.time() - DirectSolver.prnt_time) > 1.0:
        #{            
            output_string = ''
            for depth, node in enumerate(node_path_list):
            #{
                
                curr_move_str = f"{color_name(node.move[0])} ({node.move[1]})" if node.move else 'root'
                reference_move = DirectSolver.ref_moves[depth] if depth < len(DirectSolver.ref_moves) else []

                move_options, found, index = '[', False, -1
                for idx, child in enumerate(node.get_uct_children()):
                #{
                    if reference_move: found = (reference_move == child.move)
                    if idx < 5: move_options += f'{color_name(child.move[0])} ({child.move[1]}):{child.ucbonus()}/{child.value()}, '
                    if found: index = idx; break
                #}
                
                move_options = re.sub(r', $', ']', move_options)
                ref_move_str = f"{color_name(reference_move[0])} ({reference_move[1]})" if reference_move else 'past'
                if not found: output_string += (f'[{curr_move_str}]: {ref_move_str}, NOT found in {len(node.child_nodes)} moves: {move_options}\n')
                else: output_string += (f'[{curr_move_str}]: {ref_move_str}, found at {index} of {len(node.child_nodes)}: {move_options}\n')
            #}
            
            clear_output(wait=True)
            print(output_string)
            DirectSolver.prnt_time = time.time()
        #}
    #}
    
    @staticmethod
    def solve_cube(direct_cube, csolver, runtime=1800):
    #{
        start = time.time()
        solved, moves, rootnode = False, None, None
        while not solved and ((time.time() - start) < runtime):
        #{
            direct_cube, moves = csolver.cycle_solve(direct_cube, DirectSolver.order_heuristic, nmoves=30)
            caption = f"Moves: {len(moves)} : Val: {DirectSolver.generate_hstate(direct_cube)}"
            CubeView(DirectCube(direct_cube)).push_snapshot(caption=caption).draw_snapshots()
            solved, moves, rootnode = DirectSolver.direct_solve(direct_cube)
        #}
        
        return solved, moves, rootnode
    #}
    
    @staticmethod
    def direct_solve(direct_cube):
    #{
        HStateNode.reset_tree()
        rootnode = HStateNode(DirectSolver.generate_hstate(direct_cube))
        solved, node = DirectSolver.uc_tree_search(direct_cube, rootnode)
        
        moves = DirectSolver.get_move_sequence(node)
        DirectSolver.output_state_progress(solved, direct_cube, moves)
        return solved, moves, rootnode
    #}
    
    @staticmethod
    def uc_tree_search(direct_cube, rootnode, iterations=50000):
    #{
        # Catches anomoly of call on already solved cube
        if direct_cube.is_solved(): return True, rootnode
    
        # UCT loop (w/o ROLLOUT step)
        for i in range(iterations):
        #{
            node = rootnode
            cube = DirectCube(direct_cube)
            HStateNode.move_depth = 0
            
            # SELECT (a frontier leaf)
            while node.child_nodes:
                node = node.select_uct_child()
                cube.rotate(node.move)
                HStateNode.move_depth += 1
            
            # EXPAND
            solved, node = DirectSolver.expand(cube, node)
            if solved: return True, node
            elif node.child_nodes: 
                node = random.choice(node.child_nodes)

            # BACKTRACK
            while node:
                node.update()
                node = node.parent_node
        #}
        
        return False, rootnode.select_mvp_child()
    #}
    
    # @staticmethod
    # def expand(cube, node, depth=0, maxdepth=3):
    # #{
    #     for move in VectorCube.MOVES:
    #     #{
    #         if not DirectSolver.is_redundant_move(node, move):
    #             child_cube = DirectCube(cube).rotate(move)
    #             hstate = DirectSolver.generate_hstate(child_cube)

    #             if hstate.improves(node.hstate):
    #                 child_node = node.add_child(hstate, move)
    #                 if child_cube.is_solved(): return True, child_node
                    
    #             elif depth < maxdepth: DirectSolver.expand(child_cube, node)
    #     #}
        
    #     return False, node
    # #}
    
    
    @staticmethod
    def expand(cube, node):
    #{
        for move in VectorCube.MOVES:
        #{
            if not DirectSolver.is_redundant_move(node, move):
                child_cube = DirectCube(cube).rotate(move)
                hstate = DirectSolver.generate_hstate(child_cube)

                #if hstate.improves(node.hstate):
                child_node = node.add_child(hstate, move)
                if child_cube.is_solved(): return True, child_node  
        #}
        
        return False, node
    #}
    
    @staticmethod
    def is_redundant_move(node, move):
        # Aka: is-back-to-back-rotation-of-same-side
        return node.move and (node.move[0] == move[0])
    
    @staticmethod
    def get_move_sequence(leaf_node):
    #{
        moves = []
        while leaf_node:
            if leaf_node.move: moves.append(leaf_node.move)
            leaf_node = leaf_node.parent_node
        
        moves.reverse()
        return moves
    #}
    
    # Static index arrays for dance_heuristics calculations
    _rowidx1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    _rowidx2 = np.array([24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
    _cornidx1 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    _cornidx2 = np.array([27,24,25,26,31,28,29,30,35,32,33,34,39,36,37,38,43,40,41,42,47,44,45,46])

    @staticmethod
    def generate_hstate(direct_cube):
        #ftf, tcoup, consol, rcomp, cncomp = DirectSolver.dance_heuristics(direct_cube)
        #return sum(ftf), sum(tcoup), sum(consol), sum(rcomp), sum(cncomp)

        ftf, tcoup, consol = DirectSolver.dance_heuristics(direct_cube, value=False)
        return HeuristicState(sum(ftf), sum(tcoup), sum(consol))
    

    @staticmethod
    def dance_heuristics(direct_cube, value=True):
    #{
        # Face-to-face facelets
        lead_dir = direct_cube.direction_matrix[2:, DirectCube._dir_order_ed_ctr]
        foll_dir = direct_cube.direction_matrix[2:, DirectCube._dir_order_cn_ed]
        face_to_face = sum(lead_dir + foll_dir) == 0

        # Tightly coupled facelets
        lead_pos = direct_cube.facelet_matrix[2:, VectorCube._order_ed_cnt]
        foll_pos = direct_cube.facelet_matrix[2:, VectorCube._order_cn_ed]
        tightly_coupled = np.linalg.norm(foll_pos - lead_pos, axis=0) == 2

        # Consolidated pairs, rows, and corners
        consolidated = np.logical_and(face_to_face, tightly_coupled)
        #rows_complete = np.logical_and(consolidated[DirectSolver._rowidx1], consolidated[DirectSolver._rowidx2])
        #corns_complete = np.logical_and(consolidated[DirectSolver._cornidx1], consolidated[DirectSolver._cornidx2])
        
        if not value: return face_to_face, tightly_coupled, consolidated #, rows_complete, corns_complete
        else: return sum(face_to_face) + sum(tightly_coupled) + sum(consolidated)
    #}

    # Static helper variables for order_heuristic fcn
    _x_sq = np.broadcast_to(np.array([[4,0,0]]).T, (3, 72))
    _y_sq = np.broadcast_to(np.array([[0,4,0]]).T, (3, 72))
    _z_sq = np.broadcast_to(np.array([[0,0,4]]).T, (3, 72))
    I_cned = np.identity(len(VectorCube._order_cn_ed), dtype=int)

    @staticmethod
    def order_heuristic(cube):
    #{
        diff_sq = (cube.facelet_matrix[2:, VectorCube._order_cn_ed] - 
                   cube.facelet_matrix[2:, VectorCube._order_ed_cnt])**2
        return sum(sum(np.logical_or(np.logical_or((DirectSolver._x_sq == diff_sq), 
            (DirectSolver._y_sq == diff_sq)), (DirectSolver._z_sq == diff_sq))) == 3)
    #}

    @staticmethod
    def front_heuristic(direct_cube):
    #{
        flet_cn_ed  = direct_cube.facelet_matrix[2:, VectorCube._order_cn_ed]
        dlet_ed_ctr = direct_cube.direction_matrix[2:, DirectCube._dir_order_ed_ctr]
        return sum(sum(np.matmul(flet_cn_ed.T, dlet_ed_ctr) * DirectSolver.I_cned) / 2)
    #}

    @staticmethod
    def angle_heuristic(direct_cube, order=False):
    #{
        follow_pos = direct_cube.facelet_matrix[2:, VectorCube._order_cn_ed]
        lead_pos   = direct_cube.facelet_matrix[2:, VectorCube._order_ed_cnt]
        lead_dir   = direct_cube.direction_matrix[2:, DirectCube._dir_order_ed_ctr]
        difference = follow_pos - lead_pos

        projection1   = lead_pos**2 != 9
        difference_p1 = difference.T[projection1.T].reshape(-1,2).T
        lead_dir_p1   = lead_dir.T[projection1.T].reshape(-1,2).T
        rads = np.arccos(sum(np.matmul(difference_p1.T, lead_dir_p1) * 
            DirectSolver.I_cned) / np.linalg.norm(difference_p1, axis=0))
        
        # If order == True, collapse to same as order_heuristic
        if order: return sum(np.linalg.norm(difference, axis=0) == 2)
        else: return sum(np.logical_and((np.linalg.norm(difference, axis=0) == 2), (rads == 0)))
    #}
#}