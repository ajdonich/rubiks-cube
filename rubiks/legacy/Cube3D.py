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

class Cube3D:
#{
    # Always only
    NCUBELETS = 20
    CLET_KEYS = [(G,O,W),(B,O,W),(B,R,W),(G,R,W),
                 (0,O,W),(B,0,W),(0,R,W),(G,0,W),
                 (G,O,0),(B,O,0),(B,R,0),(G,R,0),
                 (G,O,Y),(B,O,Y),(B,R,Y),(G,R,Y),
                 (0,O,Y),(B,0,Y),(0,R,Y),(G,0,Y)]

    # Access these through static methods below
    _solved_state = {}
    _rotation_matrices = {}

    @staticmethod
    def clet_is_solved(cubelet):
        return sum(cubelet.state() == Cube3D.solved_state(cubelet)) == 12
    
    @staticmethod
    def get_clet_index(tri_color):
        return next(i for i in range(len(Cube3D.CLET_KEYS)) if Cube3D.CLET_KEYS[i] == tri_color) 
    
    @staticmethod
    def solved_state(cubelet):
    #{
        if not Cube3D._solved_state:
        #{
            # Model input encoding formats of solved-state per cubelet
            Cube3D._solved_state[(G,O,W)] = np.array([1,-1,1,1,0,0,0,-1,0,0,0,1])
            Cube3D._solved_state[(B,O,W)] = np.array([-1,-1,1,-1,0,0,0,-1,0,0,0,1])
            Cube3D._solved_state[(B,R,W)] = np.array([-1,1,1,-1,0,0,0,1,0,0,0,1])
            Cube3D._solved_state[(G,R,W)] = np.array([1,1,1,1,0,0,0,1,0,0,0,1])
            Cube3D._solved_state[(0,O,W)] = np.array([0,-1,1,0,0,0,0,-1,0,0,0,1])
            Cube3D._solved_state[(B,0,W)] = np.array([-1,0,1,-1,0,0,0,0,0,0,0,1])
            Cube3D._solved_state[(0,R,W)] = np.array([0,1,1,0,0,0,0,1,0,0,0,1])
            Cube3D._solved_state[(G,0,W)] = np.array([1,0,1,1,0,0,0,0,0,0,0,1])
            Cube3D._solved_state[(G,O,0)] = np.array([1,-1,0,1,0,0,0,-1,0,0,0,0])
            Cube3D._solved_state[(B,O,0)] = np.array([-1,-1,0,-1,0,0,0,-1,0,0,0,0])
            Cube3D._solved_state[(B,R,0)] = np.array([-1,1,0,-1,0,0,0,1,0,0,0,0])
            Cube3D._solved_state[(G,R,0)] = np.array([1,1,0,1,0,0,0,1,0,0,0,0])
            Cube3D._solved_state[(G,O,Y)] = np.array([1,-1,-1,1,0,0,0,-1,0,0,0,-1])
            Cube3D._solved_state[(B,O,Y)] = np.array([-1,-1,-1,-1,0,0,0,-1,0,0,0,-1])
            Cube3D._solved_state[(B,R,Y)] = np.array([-1,1,-1,-1,0,0,0,1,0,0,0,-1])
            Cube3D._solved_state[(G,R,Y)] = np.array([1,1,-1,1,0,0,0,1,0,0,0,-1])
            Cube3D._solved_state[(0,O,Y)] = np.array([0,-1,-1,0,0,0,0,-1,0,0,0,-1])
            Cube3D._solved_state[(B,0,Y)] = np.array([-1,0,-1,-1,0,0,0,0,0,0,0,-1])
            Cube3D._solved_state[(0,R,Y)] = np.array([0,1,-1,0,0,0,0,1,0,0,0,-1])
            Cube3D._solved_state[(G,0,Y)] = np.array([1,0,-1,1,0,0,0,0,0,0,0,-1])
        #}
        
        return Cube3D._solved_state[cubelet.colors]
    #}
    
    @staticmethod
    def rotation(side_angle_tuple):
    #{
        if not Cube3D._rotation_matrices:
        #{
            # Rotation matrix mappings
            for rad, deg in zip([math.pi/2, -math.pi/2, math.pi], [90, -90, 180]):
            #{
                c = round(math.cos(rad))
                s = round(math.sin(rad))

                Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

                Cube3D._rotation_matrices[(GREEN,deg)]  = Rx
                Cube3D._rotation_matrices[(BLUE,deg)]   = Rx
                Cube3D._rotation_matrices[(RED,deg)]    = Ry
                Cube3D._rotation_matrices[(ORANGE,deg)] = Ry
                Cube3D._rotation_matrices[(WHITE,deg)]  = Rz
                Cube3D._rotation_matrices[(YELLOW,deg)] = Rz
            #}
        #}
        
        return Cube3D._rotation_matrices[side_angle_tuple]
    #}
        
    # Inner class
    class Cubelet:
    #{
        def __init__(self, colors, position, orient=None):
        #{
            self.colors = colors
            self.position = position
            
            if orient is not None: self.orient = orient
            else: self.orient = np.array([[position[0],0,0],[0,position[1],0],[0,0,position[2]]]).transpose()
            
            # Defining: x_dim = 0, y_dim = 1, z_dim = 2 (i.e. position = [x,y,z])
            self._side_dim_map = { GREEN:0, BLUE:0, RED:1, ORANGE:1, WHITE:2, YELLOW:2 }
            self._side_dir_map = { GREEN:1, BLUE:-1, RED:1, ORANGE:-1, WHITE:1, YELLOW:-1 }
        #}
        
        def __repr__(self): 
            return (f"colors: {self.name()}\n"
                    f"position: {self.position}\n"
                    f"orientation:\n {self.orient}")
        
        def __eq__(self, other):
            return self.colors == other.colors
        
        def copy(self):
            return Cube3D.Cubelet(colors=self.colors, position=self.position, orient=self.orient)
        
        def matches(self, other):
            return (self == other and self.isat(other.position) and self.isorient(other.orient))
        
        def name(self):
            return (f"({color_letr(self.colors[0])},"
                    f"{color_letr(self.colors[1])},"
                    f"{color_letr(self.colors[2])})")
        
        def state(self):
            return np.concatenate((self.position, self.orient), axis=None)

        def ison(self, side):
            return self.position[self._side_dim_map[side]] == self._side_dir_map[side]
        
        def isat(self, position):
            return sum(self.position == position) == 3
                
        def isorient(self, orient):
            return sum(sum(self.orient == orient)) == 9
        
        def colorat(self, side):         
            sdim_row = self.orient[self._side_dim_map[side],:]
            return self.colors[np.flatnonzero(sdim_row == self._side_dir_map[side])[0]]
        
        def reset(self, state_v):
            state_v = np.ndarray.flatten(state_v)
            self.position = np.array(state_v[0:3])
            self.orient = np.array(state_v[3:].reshape(3,3))
        
        # NOTE: this is a direct apply, check
        # that ison(side) is client responsibility
        def apply_rotation(self, R):
            self.position = R.dot(self.position)
            self.orient = R.dot(self.orient)
    #}
    
    # Begin Cube3D implementation
    def __init__(self, copycube=None):
    #{
        if copycube is None:
        #{
            self.cubelets = []
            
            # Top layer
            self.cubelets.append(self.Cubelet(colors=(G,O,W), position=np.array([1,-1,1])))
            self.cubelets.append(self.Cubelet(colors=(B,O,W), position=np.array([-1,-1,1])))
            self.cubelets.append(self.Cubelet(colors=(B,R,W), position=np.array([-1,1,1])))
            self.cubelets.append(self.Cubelet(colors=(G,R,W), position=np.array([1,1,1])))
            self.cubelets.append(self.Cubelet(colors=(0,O,W), position=np.array([0,-1,1])))
            self.cubelets.append(self.Cubelet(colors=(B,0,W), position=np.array([-1,0,1])))
            self.cubelets.append(self.Cubelet(colors=(0,R,W), position=np.array([0,1,1])))
            self.cubelets.append(self.Cubelet(colors=(G,0,W), position=np.array([1,0,1])))

            # Middle layer
            self.cubelets.append(self.Cubelet(colors=(G,O,0), position=np.array([1,-1,0])))
            self.cubelets.append(self.Cubelet(colors=(B,O,0), position=np.array([-1,-1,0])))
            self.cubelets.append(self.Cubelet(colors=(B,R,0), position=np.array([-1,1,0])))
            self.cubelets.append(self.Cubelet(colors=(G,R,0), position=np.array([1,1,0])))

            # Bottom layer
            self.cubelets.append(self.Cubelet(colors=(G,O,Y), position=np.array([1,-1,-1])))
            self.cubelets.append(self.Cubelet(colors=(B,O,Y), position=np.array([-1,-1,-1])))
            self.cubelets.append(self.Cubelet(colors=(B,R,Y), position=np.array([-1,1,-1])))
            self.cubelets.append(self.Cubelet(colors=(G,R,Y), position=np.array([1,1,-1])))
            self.cubelets.append(self.Cubelet(colors=(0,O,Y), position=np.array([0,-1,-1])))
            self.cubelets.append(self.Cubelet(colors=(B,0,Y), position=np.array([-1,0,-1])))
            self.cubelets.append(self.Cubelet(colors=(0,R,Y), position=np.array([0,1,-1])))
            self.cubelets.append(self.Cubelet(colors=(G,0,Y), position=np.array([1,0,-1])))
        #}
        else: self.cubelets = [copylet.copy() for copylet in copycube.cubelets]
    #}
    
    def get_cubelet(self, cubelet):
        return next(clet for clet in self.cubelets if clet == cubelet)
    
    def get_cubelets(self, side):
        return [clet for clet in self.cubelets if clet.ison(side)]
    
    def get_colorat(self, side, position):
    #{
        # Cubelet will default to None if position is one of the 6 fixed centers 
        cubelet = next((clet for clet in self.cubelets if clet.isat(position)), None)
        return side if cubelet is None else cubelet.colorat(side)
    #}
    
    def numb_solved(self):
        return sum([Cube3D.clet_is_solved(clet) for clet in self.cubelets])
    
    def is_solved(self):
        return self.numb_solved() == Cube3D.NCUBELETS

    def state(self):
        return np.concatenate([cubelet.state() for cubelet in self.cubelets])
    
    def reset(self):
        for clet in self.cubelets: clet.reset(Cube3D.solved_state(clet))
      
    def scramble(self, sz=64):
    #{
        self.reset() # Start from solved/reset state
        angle = np.random.choice([90, -90, 180], size=sz)
        sides = np.random.randint(low=1, high=7, size=sz)
        for sd, ang in zip(sides, angle): self.rotate(sd,ang)
    #}
    
    def rotate(self, side, angle):
        for clet in self.get_cubelets(side): clet.apply_rotation(Cube3D.rotation((side, angle)))
#}