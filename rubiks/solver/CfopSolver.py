import random, time, itertools

from rubiks.model.CubeView import CubeView
from rubiks.model.SMAdapter import SMAdapter
from rubiks.model.CfopCube import CfopCube, cfop_name
from rubiks.model.VectorCube import VectorCube, color_name, color_letr

from rubiks.solver.UCTNode import UCTNode
from rubiks.solver.CfopState import CfopState, FullCubeState, CrossState, F2LStateOrbit, OLLState, PLLState

class CfopSolver:
#{
    ref_moves = None
    first_pass_iters = 50000
    second_pass_iters = 100000
    prnt_time = time.time()
    
    @staticmethod
    def output_state_progress(solved, solve_state, cfop_cube, moves, pair_idx=None):
    #{
        # Note: this method prints status and also applies moves to cfop_cube 
        is_f2l = isinstance(solve_state,F2LStateOrbit)
    
        if not moves: 
            print(f"Solved state \'{cfop_name(solve_state.state)}\': {solved} (zero moves applied)")
            print(f"  Numb tree nodes: {UCTNode.total_nodes}, node visits: {UCTNode.total_visits}")
            if is_f2l: print(f"  Substate: {solve_state.substate}, Pair: {pair_idx}")
            print()
        
        else:
        #{
            print(f"Solved state \'{cfop_name(solve_state.state)}\': {solved}")
            if is_f2l: print(f"  Substate: {solve_state.substate}, Pair: {pair_idx}")
            if not isinstance(solve_state, OLLState) and not isinstance(solve_state, PLLState):
                print(f"  Numb tree nodes: {UCTNode.total_nodes}, node visits: {UCTNode.total_visits}")
                print(f"  Numb moves to best cube: {len(moves)} (best scaled node value: {UCTNode.top_value})")
                print(f"  Moves: {[f'({color_letr(mv[0])}:{mv[1]})' for mv in moves]}\n")
                if not isinstance(solve_state, FullCubeState): UCTNode.reset_tree()
            
            else: print(f"  Numb moves to solution: {len(moves)}\n  Moves: {[f'({color_letr(mv[0])}:{mv[1]})' for mv in moves]}\n")
            
            view = CubeView(cfop_cube)
            for mv in moves:
                cfop_cube.rotate(mv)
                view.push_snapshot(caption=f"{color_name(mv[0])} ({mv[1]})")

            view.draw_snapshots()
        #}
        
        return cfop_cube
    #}
    
    @staticmethod
    def cfop_solve(fsolver, cfop_cube, pass1_iters=50000, pass2_iters=100000):
    #{
        CfopSolver.first_pass_iters = pass1_iters
        CfopSolver.second_pass_iters = pass2_iters

        # Solve each CFOP stage
        solved, node, moves, solve_state = CfopSolver.cross_solve(fsolver, cfop_cube)
        if solved: solved, node, moves, solve_state = CfopSolver.f2l_solve(fsolver, cfop_cube, node, moves)
        if solved: solved, moves, solve_state = CfopSolver.oll_solve(fsolver, cfop_cube, moves)
        if solved: solved, moves, solve_state = CfopSolver.pll_solve(fsolver, cfop_cube, moves)
        return solved, moves, solve_state 
    #}
    
    @staticmethod
    def cross_solve(fsolver, cfop_cube):
    #{
        solve_state = CrossState(fsolver)
        rootnode = UCTNode(solve_state.heuristic(cfop_cube))
        solved, node = CfopSolver.uc_tree_search(cfop_cube, rootnode, solve_state, iterations=CfopSolver.first_pass_iters)
        moves = CfopSolver.get_move_sequence(node)
                
        if not solved:
            # Give it a second try if it got stymied
            cube = CfopCube(cfop_cube).rotate_seq(moves)
            solved, node = CfopSolver.uc_tree_search(cube, node.set_root(), solve_state, iterations=CfopSolver.second_pass_iters)
            moves.extend(CfopSolver.get_move_sequence(node))

        cfop_cube = CfopSolver.output_state_progress(solved, solve_state, cfop_cube, moves)
        return solved, node, moves, solve_state 
    #}
    
    @staticmethod
    def f2l_solve(fsolver, cfop_cube, node, moves):
    #{
        solved = True
        solve_state = F2LStateOrbit(fsolver)
        unsolved_pairs = solve_state.get_unsolved_pairs(cfop_cube)
        if not unsolved_pairs: CfopSolver.output_state_progress(solved, solve_state, cfop_cube, [])
        
        while unsolved_pairs:
        #{            
            ci_pair = unsolved_pairs[0]
            solve_state.set_f2l_pair(ci_pair)
            while solve_state.substate != 'Done':
            #{
                iterations = CfopSolver.first_pass_iters if solve_state.substate == 'SINGLE_PASS' else CfopSolver.second_pass_iters
                solved, node = CfopSolver.uc_tree_search(cfop_cube, node.set_root(), solve_state, iterations=iterations)
                state_moves = CfopSolver.get_move_sequence(node)
                moves.extend(state_moves)
                
                cfop_cube = CfopSolver.output_state_progress(solved, solve_state, cfop_cube, state_moves, ci_pair)
                if not (solved or (solve_state.substate == 'SINGLE_PASS')): break
                else: solve_state.increment_substate()
            #}
            
            if not (solved or (solve_state.substate == 'SINGLE_PASS')): break 
            else: unsolved_pairs = solve_state.get_unsolved_pairs(cfop_cube)
        #}
        
        return solved, node, moves, solve_state 
    #}
    
    @staticmethod
    def oll_solve(fsolver, cfop_cube, moves):
    #{
        solve_state = OLLState(fsolver)
        solved, local_moves = CfopSolver.human_alg_search(cfop_cube, solve_state)
        cfop_cube = CfopSolver.output_state_progress(solved, solve_state, cfop_cube, local_moves)    
        
        moves.extend(local_moves)
        return solved, moves, solve_state 
    #}
    
    @staticmethod
    def pll_solve(fsolver, cfop_cube, moves):
    #{
        solve_state = PLLState(fsolver)
        solved, local_moves = CfopSolver.human_alg_search(cfop_cube, solve_state)
        cfop_cube = CfopSolver.output_state_progress(solved, solve_state, cfop_cube, local_moves)    
        
        moves.extend(local_moves)
        return solved, moves, solve_state 
    #}

    @staticmethod
    def human_alg_search(cfop_cube, solve_state):
    #{
        # Catch anomoly of already stage-solved cube
        if solve_state.is_complete(cfop_cube): return True, []
        
        u_rotations = ["", "U", "U2", "U\'"] if isinstance(solve_state, PLLState) else [""]
        for z_rot, u_rot, in itertools.product(["", "Y", "Y2", "Y\'"], u_rotations):
        #{
            for i, sm_name_seq in enumerate(solve_state.human_algs):
            #{
                # Human algorithm setup
                sm_cube = CfopCube(cfop_cube)
                sm_adpt = SMAdapter(sm_cube).rotate_singmaster('X2')
                if z_rot: sm_adpt.rotate_singmaster(z_rot)
                
                # Test sequence
                local_moves = []
                if u_rot: sm_adpt.rotate_singmaster(u_rot, local_moves)                
                sm_adpt.rotate_singmaster_seq(sm_name_seq, local_moves)
                if solve_state.is_complete(sm_cube): return True, local_moves 
            #}
        #}
        
        return False, [] 
    #}
    
    @staticmethod
    def uc_tree_search(cfop_cube, rootnode, solve_state, max_moves=10, iterations=50000):
    #{
        # Catches anomoly of uc_tree_search called on already stage-solved cube
        if solve_state.is_complete(cfop_cube): return True, rootnode
    
        # UCT loop (w/o MC rollout)
        for i in range(iterations):
        #{
            node = rootnode
            cube = CfopCube(cfop_cube)
            UCTNode.move_depth = 0
            
            # SELECT (down to a leaf node on frontier)
            node_path_debug = [node]
            while node.child_nodes:
                node = node.select_uct_child()
                cube.rotate(node.move)
                UCTNode.move_depth += 1
                node_path_debug.append(node)
            
            # EXPAND (adds some subset of the 18 possible rotations)
            if UCTNode.move_depth < max_moves:
            #{
                for move in solve_state.expand_moves(cube):
                #{
                    if not CfopSolver.is_redundant_move(node, move):
                        child_cube = CfopCube(cube).rotate(move)
                        child_node = node.add_child(solve_state.heuristic(child_cube), move)
                        if solve_state.is_complete(child_cube): return True, child_node
                #}

                # Descend (one step, no real ROLLOUT)
                node = random.choice(node.child_nodes)
            #}
            
            # BACKTRACK
            while node:
                node.update()
                node = node.parent_node
        #}
        
        return False, rootnode.select_mvp_child()
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
#}