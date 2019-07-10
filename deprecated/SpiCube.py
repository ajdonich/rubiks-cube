import math
import numpy as np

# COLORS/SIDES:
WHITE, W  = 1, 1
ORANGE, O = 2, 2
GREEN, G  = 3, 3
RED, R    = 4, 4
BLUE, B   = 5, 5
YELLOW, Y = 6, 6

# COLORS/SIDES in order
SIDES = [WHITE, ORANGE, GREEN, RED, BLUE, YELLOW]

# Print convenience maps/fcns
color_letr_map = { 0:0, WHITE:'W', ORANGE:'O', GREEN:'G', RED:'R', BLUE:'B', YELLOW:'Y' }
color_name_map = { 0:'N/A', WHITE:'WHITE', ORANGE:'ORANGE', GREEN:'GREEN', RED:'RED', BLUE:'BLUE', YELLOW:'YELLOW' }

def color_letr(fc): return color_letr_map[fc]
def color_name(fc): return color_name_map[fc]
def tri_color_name(tc): return (f"({color_letr(tc[0])},{color_letr(tc[1])},{color_letr(tc[2])})")

class Facelet:
#{
    def __init__(self, color, index, position):
        self.color = color
        self.index = index
        self.position = position

    def __repr__(self):
        return f"{color_name(self.color)}-{self.index}"

    def __eq__(self, other):
        return (self.color == other.color) and (self.position == other.position)

    def matches(self, color, index):
        return (self.color == color) and (self.index == index)
    
    def copy(self):
        return Facelet(color=self.color, index=self.index, position=self.position)

    def name(self):
        return f"{color_name(self.color)}-{self.index} : {self.position}"

    def state(self):
        return np.concatenate(([self.color], [self.index], self.position))

    def dot(self, other):
        return self.position.dot(other.position)

    def isat(self, position):
        return sum(self.position == position) == 3
    
    # NO check for isonside(...)
    def apply_rotation(self, R):
        self.position = R.dot(self.position)
#}

class Spiderweb:
#{
    def __init__(self, flet_a, flet_b):
        self.flet_a = flet_a
        self.flet_b = flet_b
    
    def __repr__(self):
        return self.flet_a.__repr__() + " to " + self.flet_b.__repr__()
    
    def distance(self):
        return np.linalg.norm(self.flet_b.position - self.flet_a.position) #- self.d_factor
#}

class SpiCube:
#{
    # Solved facelet position vectors, ordered left-to-right top-to-bottom per side
    SOLVED_POS = { WHITE:   np.array([[-2,-2,3],[-2,0,3],[-2,2,3],[0,-2,3],[0,0,3],[0,2,3],[2,-2,3],[2,0,3],[2,2,3]]),
                   ORANGE:  np.array([[-2,-3,2],[0,-3,2],[2,-3,2],[-2,-3,0],[0,-3,0],[2,-3,0],[-2,-3,-2],[0,-3,-2],[2,-3,-2]]),
                   GREEN:   np.array([[3,-2,2],[3,0,2],[3,2,2],[3,-2,0],[3,0,0],[3,2,0],[3,-2,-2],[3,0,-2],[3,2,-2]]),
                   RED:     np.array([[2,3,2],[0,3,2],[-2,3,2],[2,3,0],[0,3,0],[-2,3,0],[2,3,-2],[0,3,-2],[-2,3,-2]]),
                   BLUE:    np.array([[-3,2,2],[-3,0,2],[-3,-2,2],[-3,2,0],[-3,0,0],[-3,-2,0],[-3,2,-2],[-3,0,-2],[-3,-2,-2]]),
                   YELLOW:  np.array([[2,-2,-3],[2,0,-3],[2,2,-3],[0,-2,-3],[0,0,-3],[0,2,-3],[-2,-2,-3],[-2,0,-3],[-2,2,-3]]) }


    # Fixed cube facelet centers
    CENTERS = { WHITE:  Facelet(color=WHITE,  index=4, position=np.array([ 0,  0,  3])),
                ORANGE: Facelet(color=ORANGE, index=4, position=np.array([ 0, -3,  0])),
                GREEN:  Facelet(color=GREEN,  index=4, position=np.array([ 3,  0,  0])),
                RED:    Facelet(color=RED,    index=4, position=np.array([ 0,  3,  0])),
                BLUE:   Facelet(color=BLUE,   index=4, position=np.array([-3,  0,  0])),
                YELLOW: Facelet(color=YELLOW, index=4, position=np.array([ 0,  0, -3])) }
    
    # The 18 possible moves from any cube state
    MOVES = [(sd,ang) for sd in SIDES for ang in [90,-90,180]]
    
    # Access through static methods below
    _rotation_matrices = {}
    
    @staticmethod
    def rotation(side_angle_tuple):
    #{
        if not SpiCube._rotation_matrices:
        #{
            # Rotation matrix mappings
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                SpiCube._rotation_matrices[(GREEN,deg)]  = Rx
                SpiCube._rotation_matrices[(BLUE,deg)]   = Rx
                SpiCube._rotation_matrices[(RED,deg)]    = Ry
                SpiCube._rotation_matrices[(ORANGE,deg)] = Ry
                SpiCube._rotation_matrices[(WHITE,deg)]  = Rz
                SpiCube._rotation_matrices[(YELLOW,deg)] = Rz
            #}
        #}
        
        return SpiCube._rotation_matrices[side_angle_tuple]
    #}
    
    # Dist scale factor
    _DFACTOR = 1.0
    
    # Begin SpiCube implementation
    def __init__(self, copycube=None):
    #{
        # Construct facelets
        if copycube is not None: self.facelets = [copylet.copy() for copylet in copycube.facelets]
        else: self.facelets = [Facelet(side, ind, pos) for side in SIDES for ind, pos in 
                               zip(range(9), SpiCube.SOLVED_POS[side]) if (ind != 4)]

        # Connect webs
        self.spiderwebs = []
        for flet in self.facelets:
        #{
            # Connect corner and edge facelets to fixed center facelet
            self.spiderwebs.append(Spiderweb(flet, SpiCube.CENTERS[flet.color]))

            # Connect corner facelets to their adjacent edge facelets
            if flet.index == 0:
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 1)))
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 3)))
            if flet.index == 2:
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 1)))
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 5)))
            if flet.index == 6:
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 3)))
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 7)))
            if flet.index == 8:
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 5)))
                self.spiderwebs.append(Spiderweb(flet, self.get_facelet(flet.color, 7)))   
        #}
        
        # Lazy init upon first cube constructed
        if SpiCube._DFACTOR == 1.0: SpiCube._DFACTOR = self.distance()
    #}
    
    def isinlayer(self, flet, side):
        return flet.dot(SpiCube.CENTERS[side]) > 0
    
    def isonside(self, flet, side):
        return flet.dot(SpiCube.CENTERS[side]) >= 9
    
    def get_facelets(self, side, layer=True):
        if layer: return [flet for flet in self.facelets if self.isinlayer(flet, side)]
        else:     return [flet for flet in self.facelets if self.isonside(flet, side)]

    def get_facelet(self, color, index):
        return next(flet for flet in self.facelets if flet.matches(color, index))
    
    def get_colorat(self, side, position):
    #{
        # Facelet will default to None if position is one of the 6 fixed centers 
        facelet = next((flet for flet in self.facelets if flet.isat(position)), None)
        return facelet.color if facelet is not None else side
    #}
    
#     def numb_solved(self):
#         return sum([Cube3D.clet_is_solved(clet) for clet in self.cubelets])
    
    def is_solved(self):
        return self.distance() == 1.0

    def state(self):
        return np.concatenate([flet.state() for flet in self.facelets])
    
    def reset(self):
        for flet in self.facelets: flet.position = SpiCube.SOLVED_POS[flet.color][flet.index]
            
    def distance(self):
        return sum([web.distance() for web in self.spiderwebs]) / SpiCube._DFACTOR
    
    def scramble(self, sz=64):
    #{
        self.reset() # Start from solved/reset state
        angle = np.random.choice([90, -90, 180], size=sz)
        sides = np.random.randint(low=1, high=7, size=sz)
        for move in zip(sides, angle): self.rotate(move)
        return self
    #}
    
    def rotate(self, move):
        for flet in self.get_facelets(side=move[0]): flet.apply_rotation(SpiCube.rotation(move))
        return self
#}