class CubeSolver:
#{
    def __init__(self):
        self.attempts = 0
        self.high_score = 0

    # Possible moves convenience lists
    dl = [CLOCKWISE, COUNTER_CLOCK]
    sl = [WHITE, ORANGE, GREEN, RED, BLUE, YELLOW]
    
    def solve_a(self, cube, move_stack=None):
    #{
        #### Debug Metrics ##
#         self.attempts += 1
#         score = cube.heuristic_a()
        
#         if score > self.high_score:
#             self.high_score = score
#             print("At attempt: ", solver.attempts)
#             print("New high score of: ", solver.high_score)
#             cube.draw_projection()
        
#         if (solver.attempts % 50000) == 0:
#             print("Attempted: ", solver.attempts)
        #####################
      
        # Cube not solved while heuristic_a() < 54
        # Ideally solves it in like 30 moves or less
    
        if move_stack == None: move_stack = [cube]
        else: move_stack.append(cube)

        if cube.heuristic_a() == 54: return move_stack
        elif len(move_stack) > 30: return None
        
        next_moves = sorted([cube.rotate(sd, dr, copy_a=True) for dr in self.dl for sd in self.sl], reverse=True)
        
        solution = None
        for mv in next_moves:
        #{
            # Move must be distinct from every previous move-state in its history
            if sum([mv != past_mv for past_mv in move_stack]) == len(move_stack):
            #{    
                solution = self.solve_a(mv, move_stack.copy())
                if solution is not None: break
            #}
        #}
        
        return solution
    #}
#}