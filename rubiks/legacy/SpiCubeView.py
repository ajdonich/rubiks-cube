import matplotlib
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.colors as mpc

# Plotting color map
color_plot_map = { WHITE: mcd.CSS4_COLORS['ivory'],
                   ORANGE: mcd.XKCD_COLORS['xkcd:orange'].upper(), 
                   GREEN: mcd.CSS4_COLORS['green'],
                   RED: mcd.XKCD_COLORS['xkcd:crimson'].upper(),
                   BLUE: mcd.XKCD_COLORS['xkcd:blue'].upper(),
                   YELLOW: mcd.XKCD_COLORS['xkcd:goldenrod'].upper() }

def color(fc): return color_plot_map[fc]

class SpiCubeView:
#{
    OFFSET     = 130
    CENTERS    = np.array([[0,0,3],[0,-3,0],[3,0,0],[0,3,0],[-3,0,0],[0,0,-3]])
    ANCHOR_POS = { WHITE:(40,70), ORANGE:(10,40), GREEN:(40,40), RED:(70,40), BLUE:(100,40), YELLOW:(40,10) }
    
    
    def __init__(self, scube):
        self.spider_cube = scube
        self.patch_sequence = [[]]
        self.caption_sequence = [[]]
    
    def reset_snapshots(self):
        self.patch_sequence = []
        self.caption_sequence = []
    
    def iscenter(self, position):
        return next((True for cent in self.CENTERS if sum(cent == position) == 3), False)
    
    def get_plot_rects(self, faces, anchor=(0,0), mask=None):
    #{    
        rects = []
        for r in range(3):
            for c in range(3):
                
                x = anchor[0] + (c*10)
                y = anchor[1] + 20  - (r*10)
                clr = color(faces[r,c])
                
                if mask is not None and (mask[r,c] == 'gray'):
                    g = np.mean(mpc.to_rgb(clr)) * 0.667
                    clr = mpc.to_hex((g,g,g))
                  
                rects.append(plt.Rectangle((x, y), 10, 10, fc=clr))

        return rects
    #}
    
    def get_gridlines(self, seqnumb=0):
    #{
        lines = []            
        for anchor in [(40,70), (10,40), (40,40), (70,40), (100,40), (40,10)]:
        #{
            xoff = self.OFFSET * seqnumb
        
            for r in range(4):
                y = anchor[1] + (r*10)
                lines.append(plt.Line2D((anchor[0] + xoff, anchor[0] + xoff + 30), (y, y), lw=1, color='k'))

            for c in range(4):
                x = anchor[0] + xoff + (c*10)
                lines.append(plt.Line2D((x, x), (anchor[1], anchor[1] + 30), lw=1, color='k'))
        #}
        
        return lines
    #}
    
    def create_patches(self, cubelet=None, seqnumb=0):
    #{
        rects = []
        mask = None
        
        for side in SIDES:
            anchor = (self.ANCHOR_POS[side][0] + (self.OFFSET * seqnumb), self.ANCHOR_POS[side][1])
            faces = [self.spider_cube.get_colorat(side, pos) for pos in SpiCube.SOLVED_POS[side]]
            rects.extend(self.get_plot_rects(np.reshape(faces, (3,3)), anchor, mask))
        
        return rects
    #}
    
    # Displays a single cube projection
    def draw_projection(self, cubelet=None):
    #{
        fig = plt.figure(figsize=[4, 3])
        ax = fig.add_axes([0, 0, 1, 1])
        
        rects = self.create_patches(cubelet)
        for r in rects: ax.add_patch(r)
        for ln in self.get_gridlines(): ax.add_line(ln)
 
        ax.axis('scaled')
        ax.axis('off')
        plt.show()
    #}
    
    def push_snapshot(self, cubelet=None, caption=""):
    #{
        # Adjust to 5 cube images per row
        top = len(self.patch_sequence)-1
        if len(self.patch_sequence[top]) > 4:
            self.caption_sequence.append([])
            self.patch_sequence.append([])
            top = len(self.patch_sequence)-1

        seqnumb = len(self.patch_sequence[top])
        rects = self.create_patches(cubelet, seqnumb)
        self.patch_sequence[top].append(rects)
        self.caption_sequence[top].append(caption)
    #}
    
    # Displays a move sequence
    def draw_snapshops(self):
    #{
        for row in range(len(self.patch_sequence)):
        #{
            nmoves = len(self.patch_sequence[row])
            width = (4 * nmoves) + (nmoves - 1)
            fig = plt.figure(figsize=[width, 3])
            ax = fig.add_axes([0, 0, 1, 1])

            for seqnumb in range(len(self.patch_sequence[row])):
            #{
                rects = self.patch_sequence[row][seqnumb]
                caption = self.caption_sequence[row][seqnumb]

                for r in rects: ax.add_patch(r)
                for ln in self.get_gridlines(seqnumb): ax.add_line(ln)
                plt.text(20 +(self.OFFSET * seqnumb), 110, caption)
            #}

            ax.axis('scaled')
            ax.axis('off')
        #}
        
        plt.show()
    #}
#}