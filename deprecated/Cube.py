import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

# Color := Side number definitions:
WHITE  = 1
ORANGE = 2
GREEN  = 3
RED    = 4
BLUE   = 5
YELLOW = 6

color_name_map = { WHITE:  'WHITE',
                   ORANGE: 'ORANGE', 
                   GREEN:  'GREEN',
                   RED:    'RED',
                   BLUE:   'BLUE',
                   YELLOW: 'YELLOW' }

color_letr_map = { WHITE:  'W',
                   ORANGE: 'O', 
                   GREEN:  'G',
                   RED:    'R',
                   BLUE:   'B',
                   YELLOW: 'Y' }

# 'Ring' rotation directions:
CLOCKWISE = 'clockwise'
COUNTER_CLOCK = 'counter_clockwise'

# Row or col of any Side
TOP = 'top'
BOT = 'bottom'
LFT = 'left'
RHT = 'right'

color_plot_map = { WHITE: mcd.CSS4_COLORS['ivory'],
                   ORANGE: mcd.XKCD_COLORS['xkcd:orange'].upper(), 
                   GREEN: mcd.CSS4_COLORS['green'],
                   RED: mcd.XKCD_COLORS['xkcd:crimson'].upper(),
                   BLUE: mcd.XKCD_COLORS['xkcd:blue'].upper(),
                   YELLOW: mcd.XKCD_COLORS['xkcd:goldenrod'].upper() }

def color(fc): return color_plot_map[fc]
def color_name(fc): return color_name_map[fc]
def color_letr(fc): return color_letr_map[fc]

class Cube:
#{
    class Side:
    #{
        def __init__(self, color):
            self.faces = np.ones((3,3), dtype=int) * color
        
        def __repr__(self):
            return (f"[[{color_letr(self.faces[0,0])} {color_letr(self.faces[0,1])} {color_letr(self.faces[0,2])}]\n"
                    f"  [{color_letr(self.faces[1,0])} {color_letr(self.faces[1,1])} {color_letr(self.faces[1,2])}]\n"
                    f"  [{color_letr(self.faces[2,0])} {color_letr(self.faces[2,1])} {color_letr(self.faces[2,2])}]]")
        
        def __eq__(self, other):
            return np.array_equal(self.faces, other.faces)
        
        def rotate_master(self, direction):
            
            if direction == CLOCKWISE:       self.faces = np.rot90(self.faces, axes=(1,0))
            elif direction == COUNTER_CLOCK: self.faces = np.rot90(self.faces, axes=(0,1))
            else: print(f"Invalid rotation direction {direction}")
        
        def set_edge(self, location, edge=np.zeros((3,), dtype=int), invert=False):
        #{
            swapped_edge = None
            if invert: edge = edge[::-1]
        
            if location == 'top':
                swapped_edge = np.copy(self.faces[0,:])
                self.faces[0,:] = edge
            elif location == 'bottom':
                swapped_edge = np.copy(self.faces[2,:])
                self.faces[2,:] = edge
            elif location == 'left':
                swapped_edge = np.copy(self.faces[:,0])
                self.faces[:,0] = edge
            elif location == 'right':
                swapped_edge = np.copy(self.faces[:,2])
                self.faces[:,2] = edge
            else: print(f"Invalid edge location: {location}")
        
            return swapped_edge
        #}
        
        def get_plot_rects(self, anchor=(0,0)):
        #{        
            rects = []
            for r in range(3):
                for c in range(3):
                    
                    x = anchor[0] + (c*10)
                    y = anchor[1] + 20  - (r*10)
                    clr = color(self.faces[r,c])
                    rects.append(plt.Rectangle((x, y), 10, 10, fc=clr))
            
            return rects
        #}
    #} End Inner Class Side

    def __init__(self):
        self.sides = {1: self.Side(WHITE), 
                      2: self.Side(ORANGE), 
                      3: self.Side(GREEN), 
                      4: self.Side(RED), 
                      5: self.Side(BLUE), 
                      6: self.Side(YELLOW)}
    
    def __lt__(self, other):
        return self.heuristic_a() < other.heuristic_a()
    
    def __eq__(self, other):        
        return (self.sides[WHITE]  == other.sides[WHITE] and 
                self.sides[ORANGE] == other.sides[ORANGE] and 
                self.sides[GREEN]  == other.sides[GREEN] and
                self.sides[RED]    == other.sides[RED] and
                self.sides[BLUE]   == other.sides[BLUE] and
                self.sides[YELLOW] == other.sides[YELLOW])
        
    def heuristic_a(self):
    #{
        total = 0
        for fcolor, side in self.sides.items(): 
            total += sum([np.sum(np.equal(f, fcolor)) for f in side.faces])
        
        return total
    #}
    
    def heuristic_top_layer(self):
    #{
        total = 0
        for fcolor, side in self.sides.items():
            total += sum([np.sum(np.equal(f, fcolor)) for f in side.faces])
        
        return total
    #}

    def scramble(self, sz=64):
    #{
        directions = np.random.random(size=sz)
        sides = np.random.randint(low=1, high=7, size=sz)
        
        for sd, dr in zip(sides, directions):
            self.rotate(sd, CLOCKWISE if dr < 0.5 else COUNTER_CLOCK)
    #}
    
    # The sole, base state-transition fcn
    def rotate(self, side, direction, copy_a=False):
    #{
        if copy_a: return copy.deepcopy(self).rotate(side, direction)
        
        self.sides[side].rotate_master(direction)
        
        if side == WHITE:
            if direction == CLOCKWISE:
                wrap = self.sides[GREEN].set_edge(TOP)
                wrap = self.sides[ORANGE].set_edge(TOP, wrap)
                wrap = self.sides[BLUE].set_edge(TOP, wrap)
                wrap = self.sides[RED].set_edge(TOP, wrap)
                wrap = self.sides[GREEN].set_edge(TOP, wrap)

            else:
                wrap = self.sides[GREEN].set_edge(TOP)
                wrap = self.sides[RED].set_edge(TOP, wrap)
                wrap = self.sides[BLUE].set_edge(TOP, wrap)
                wrap = self.sides[ORANGE].set_edge(TOP, wrap)
                wrap = self.sides[GREEN].set_edge(TOP, wrap)
        
        elif side == ORANGE:
            if direction == CLOCKWISE:
                wrap = self.sides[GREEN].set_edge(LFT)
                wrap = self.sides[YELLOW].set_edge(LFT, wrap)
                wrap = self.sides[BLUE].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[WHITE].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[GREEN].set_edge(LFT, wrap)

            else:
                wrap = self.sides[GREEN].set_edge(LFT)
                wrap = self.sides[WHITE].set_edge(LFT, wrap)
                wrap = self.sides[BLUE].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[YELLOW].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[GREEN].set_edge(LFT, wrap)
          
        elif side == GREEN:
            if direction == CLOCKWISE:
                wrap = self.sides[WHITE].set_edge(BOT)
                wrap = self.sides[RED].set_edge(LFT, wrap)
                wrap = self.sides[YELLOW].set_edge(TOP, wrap, invert=True)
                wrap = self.sides[ORANGE].set_edge(RHT, wrap)
                wrap = self.sides[WHITE].set_edge(BOT, wrap, invert=True)

            else:
                wrap = self.sides[WHITE].set_edge(BOT)
                wrap = self.sides[ORANGE].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[YELLOW].set_edge(TOP, wrap)
                wrap = self.sides[RED].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[WHITE].set_edge(BOT, wrap)
        
        elif side == RED:
            if direction == CLOCKWISE:
                wrap = self.sides[GREEN].set_edge(RHT)
                wrap = self.sides[WHITE].set_edge(RHT, wrap)
                wrap = self.sides[BLUE].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[YELLOW].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[GREEN].set_edge(RHT, wrap)
            
            else:
                wrap = self.sides[GREEN].set_edge(RHT)
                wrap = self.sides[YELLOW].set_edge(RHT, wrap)
                wrap = self.sides[BLUE].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[WHITE].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[GREEN].set_edge(RHT, wrap)
            
        elif side == BLUE:
            if direction == CLOCKWISE:
                wrap = self.sides[WHITE].set_edge(TOP)
                wrap = self.sides[ORANGE].set_edge(LFT, wrap, invert=True)
                wrap = self.sides[YELLOW].set_edge(BOT, wrap)
                wrap = self.sides[RED].set_edge(RHT, wrap, invert=True)
                wrap = self.sides[WHITE].set_edge(TOP, wrap)

            else:
                wrap = self.sides[WHITE].set_edge(TOP)
                wrap = self.sides[RED].set_edge(RHT, wrap)
                wrap = self.sides[YELLOW].set_edge(BOT, wrap, invert=True)
                wrap = self.sides[ORANGE].set_edge(LFT, wrap)
                wrap = self.sides[WHITE].set_edge(TOP, wrap, invert=True)
            
        elif side == YELLOW:
            if direction == CLOCKWISE: 
                wrap = self.sides[GREEN].set_edge(BOT)
                wrap = self.sides[RED].set_edge(BOT, wrap)
                wrap = self.sides[BLUE].set_edge(BOT, wrap)
                wrap = self.sides[ORANGE].set_edge(BOT, wrap)
                wrap = self.sides[GREEN].set_edge(BOT, wrap)
            
            else:
                wrap = self.sides[GREEN].set_edge(BOT)
                wrap = self.sides[ORANGE].set_edge(BOT, wrap)
                wrap = self.sides[BLUE].set_edge(BOT, wrap)
                wrap = self.sides[RED].set_edge(BOT, wrap)
                wrap = self.sides[GREEN].set_edge(BOT, wrap)

        else: print(f"Invalid rotation side {side}")
        return self
    #}
    
    ######## Move fcns below to an MVC View Class 
    
    def get_gridlines(self):
    #{
        lines = []            
        for anchor in [(40,70), (10,40), (40,40), (70,40), (100,40), (40,10)]:
        #{
            for r in range(4):
                y = anchor[1] + (r*10)
                lines.append(plt.Line2D((anchor[0], anchor[0] + 30), (y, y), lw=1, color='k'))

            for c in range(4):
                x = anchor[0] + (c*10)
                lines.append(plt.Line2D((x, x), (anchor[1], anchor[1] + 30), lw=1, color='k'))
        #}
        
        return lines
    #}
    
    def draw_projection(self):
    #{
        rects = []
        rects.extend(self.sides[WHITE].get_plot_rects((40,70)))
        rects.extend(self.sides[ORANGE].get_plot_rects((10,40)))
        rects.extend(self.sides[GREEN].get_plot_rects((40,40)))
        rects.extend(self.sides[RED].get_plot_rects((70,40)))
        rects.extend(self.sides[BLUE].get_plot_rects((100,40)))
        rects.extend(self.sides[YELLOW].get_plot_rects((40,10)))
        
        fig = plt.figure(figsize=[4, 3])
        ax = fig.add_axes([0, 0, 1, 1])
        
        for r in rects: ax.add_patch(r)
        for ln in self.get_gridlines(): ax.add_line(ln)
 
        ax.axis('scaled')
        ax.axis('off')
        plt.show()
    #}
#}
