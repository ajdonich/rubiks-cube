import math
import numpy as np

class UCTNode:
#{
    UCTFACTOR = math.sqrt(2.0)
    
    total_visits = 0
    total_nodes = 0
    move_depth = 0
    top_value = 0
    
    # These for derivation of incremental mean 
    # and standard deviation of heuristic values
    mu_h = 0
    sn_h = 0
    sigma_h = 0
    n = 0
        
    @classmethod
    def update_sigma(cls, heuristic):
    #{
        cls.n += 1
        mu_prev = cls.mu_h
        sn_prev = cls.sn_h
        
        cls.mu_h = mu_prev + ((heuristic - mu_prev) / cls.n)
        cls.sn_h = sn_prev + ((heuristic - mu_prev) * (heuristic - cls.mu_h))
        cls.sigma_h = math.sqrt(cls.sn_h/cls.n)
    #}

    @classmethod
    def reset_tree(cls):
    #{
        cls.total_visits = 0
        cls.total_nodes = 0
        cls.move_depth = 0
        cls.top_value = 0
    #}
    
    # Note: scale/shift should map heuristic to a range of approximately (0,7) as cube 
    # moves from utterly scrambled (0) to solved (7). (Inferred by experiment for an
    # efficient balance of exploit/explore of tree; assumes UCTFACTOR = √2 as above.)
    def __init__(self, heuristic, scale=-30.0, shift=162, move=None, parent=None):
    #{
        # Move that "produces" this node/cube-state
        self.move = move

        # Defaults: -30 and 162 for Min-Dist-Home heuristic
        self.heuristic = (heuristic - shift) / scale

        # Was just used to aid hyperperameter tuning
        # UCTNode.update_sigma(self.heuristic)
        
        self.numb_visits = 0
        self.child_nodes = []
        self.parent_node = parent
        
        UCTNode.total_nodes += 1
    #}
    
    def __repr__(self):
        return (f"Move:     {self.move}\n"
                f"Visits:   {self.numb_visits}\n"
                f"Value:    {self.heuristic}\n")
    
    def set_root(self):
        self.move = None
        self.parent_node = None
        return self
        
    def ucbonus(self):
    #{
        # Calc uncertainty bonus
        exploit = (1 + self.numb_visits)
        explore = np.log(self.parent_node.numb_visits)
        return UCTNode.UCTFACTOR * math.sqrt(explore/exploit)
    #}

    def select_uct_child(self):
        return max(self.child_nodes, key=lambda cld: self.heuristic + cld.ucbonus())
    
    def get_uct_children(self):
        return sorted(self.child_nodes, key=lambda cld: self.heuristic + cld.ucbonus())
    
    def select_mvp_child(self, mvp=None, depth=0):
    #{
        depth += 1 # Depth here is just for algorithm diagnosics 
        if depth > UCTNode.move_depth: UCTNode.move_depth = depth

        for child in self.child_nodes: mvp = child.select_mvp_child(mvp, depth)
        if (mvp is None) or (self.heuristic > mvp.heuristic): mvp = self
        return mvp
    #}
    
    def add_child(self, heuristic, move):
        child = UCTNode(heuristic, move=move, parent=self)
        if child.heuristic > UCTNode.top_value:
            UCTNode.top_value = child.heuristic

        self.child_nodes.append(child)
        return child
    
    def update(self):
        self.numb_visits += 1
        UCTNode.total_visits += 1
#}

class HStateNode(UCTNode):
#{
    # Defaults: 30 and 0 for dance heuristic. For order heuristic use 10 and 0. 
    def __init__(self, hstate, scale=30.0, shift=0, move=None, parent=None):
        super().__init__(hstate.value, scale, shift, move, parent)
        self.hstate = hstate

    def __repr__(self):
    #{
        return (f"Move:     {self.move}\n"
                f"Visits:   {self.numb_visits}\n"
                f"Value:    {self.hstate.value}\n"
                f"  Ftf:    {self.hstate.face_to_face}\n"
                f"  TCoup:  {self.hstate.tight_couple}\n"
                f"  Consol: {self.hstate.consolidated}\n")
    #}              
                    
    def add_child(self, hstate, move):
    #{
        child = HStateNode(hstate, move=move, parent=self)
        if child.heuristic > UCTNode.top_value:
            UCTNode.top_value = child.heuristic

        self.child_nodes.append(child)
        return child
    #}
#}

class HeuristicState:
#{
    def __init__(self, ftf, tcoup, consol):
            self.face_to_face = ftf
            self.tight_couple = tcoup
            self.consolidated = consol
            self.value = ftf + tcoup + consol

            #self.rows_complete   = 0
            #self.corns_complete  = 0

    def improves(self, other, mode=0):
        if mode == 0: return (self.value > other.value)
        elif mode == 1: return (self.face_to_face > other.face_to_face)
        elif mode == 2: return (self.tight_couple > other.tight_couple)
        elif mode == 3: return (self.consolidated > other.consolidated)
        elif mode == 4: return (self.face_to_face > other.face_to_face) and (self.value > other.value)
        else: assert False, "Invalid mode"

        # if ambig: return ((self.face_to_face > other.face_to_face) or (self.tight_couple > other.tight_couple))
        # else: return (((self.face_to_face > other.face_to_face) and (self.tight_couple >= other.tight_couple)) or
        #               ((self.face_to_face >= other.face_to_face) and (self.tight_couple > other.tight_couple)))

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other): 
        return ((self.face_to_face == other.face_to_face) and 
                (self.tight_couple == other.tight_couple))

    def __repr__(self):
        return f"{self.value}/{self.face_to_face}/{self.tight_couple}"
#}