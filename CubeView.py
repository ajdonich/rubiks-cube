import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib._color_data as mcd

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Observer import Observer
from SMAdapter import SMAdapter
from VectorCube import VectorCube, SIDES, WHITE_CB, W, ORANGE_CB, O, GREEN_CB, G, RED_CB, R, BLUE_CB, B, YELLOW_CB, Y

# Plotting color map
color_plot_map = { WHITE_CB:  mcd.CSS4_COLORS['ivory'],
                   ORANGE_CB: mcd.XKCD_COLORS['xkcd:orange'].upper(), 
                   GREEN_CB:  mcd.CSS4_COLORS['green'],
                   RED_CB:    mcd.XKCD_COLORS['xkcd:crimson'].upper(),
                   BLUE_CB:   mcd.XKCD_COLORS['xkcd:blue'].upper(),
                   YELLOW_CB: mcd.XKCD_COLORS['xkcd:goldenrod'].upper() }

def color(fc): return color_plot_map[fc]

class CubeView(Observer):
#{
    OFFSET     = 130
    CENTERS    = np.array([[0,0,3],[0,-3,0],[3,0,0],[0,3,0],[-3,0,0],[0,0,-3]])
    ANCHOR_POS = { WHITE_CB:(40,70), ORANGE_CB:(10,40), GREEN_CB:(40,40), RED_CB:(70,40), BLUE_CB:(100,40), YELLOW_CB:(40,10) }

    # Access through static methods below
    _poly_verts = None
    _divider_verts = None
    
    @staticmethod
    def get_poly_verts():
    #{
        if CubeView._poly_verts is None:
        #{
            verts_list = []
            for flet in VectorCube._facelet_matrix.T:
            #{
                dims = np.nonzero(flet[2:]**2 != 9)[0]
                verts = np.meshgrid(flet[2:], [1,1,1,1])[0]
                verts_list.append(verts)

                for i, v in enumerate(verts):
                    if i == 0:   v[dims[0]] -= 1; v[dims[1]] -= 1
                    elif i == 1: v[dims[0]] -= 1; v[dims[1]] += 1
                    elif i == 2: v[dims[0]] += 1; v[dims[1]] += 1
                    elif i == 3: v[dims[0]] += 1; v[dims[1]] -= 1
            #}
            CubeView._poly_verts = np.array(verts_list)
        #}

        return CubeView._poly_verts
    #}
    
    # @staticmethod
    # def get_divider_verts():
    # #{
    #     if CubeView._divider_verts is None:
    #     #{
    #         verts_list = []
    #         for flet in VectorCube._facelet_matrix[:, VectorCube._centers].T:
    #         #{
    #             divider = (flet[2:]/3).astype(int)
    #             dims = np.nonzero(flet[2:]**2 != 9)[0]
    #             verts = np.meshgrid(divider, [1,1,1,1])[0]
    #             verts_list.append(verts)

    #             for i, v in enumerate(verts):
    #                 if i == 0:   v[dims[0]] -= 2; v[dims[1]] -= 2
    #                 elif i == 1: v[dims[0]] -= 2; v[dims[1]] += 2
    #                 elif i == 2: v[dims[0]] += 2; v[dims[1]] += 2
    #                 elif i == 3: v[dims[0]] += 2; v[dims[1]] -= 2
    #         #}
            
    #         # First 6 are "lateral" dividers, last 6 are the "medial" dividers
    #         CubeView._divider_verts = np.concatenate((verts_list, verts_list))
    #     #}

    #     return CubeView._divider_verts
    # #}

    def __init__(self, cube_a=None, sm_adapter=None):
    #{
        self.viewable_cube = cube_a
        self.sm_adapter = sm_adapter
        if self.sm_adapter is not None:
            self.viewable_cube = self.sm_adapter.local_cube

        self.reset_snapshots()
        self.reset_animation()
    #}
    
    def reset_snapshots(self):
        self.patch_sequence = [[]]
        self.caption_sequence = [[]]
    
    def reset_animation(self):
        self.ax_3d = None
        self.fig_3d = None
        self.fpoly_stack = []
        self.anim_zip_stack = []
        self.divider_stack = []
    
    def press_record(self, stop_recording=False):
    #{
        if stop_recording: self.sm_adapter.unregister_observer(self)
        else: self.sm_adapter.register_observer(self)

        if not self.fig_3d:
            self.fig_3d = plt.figure()
            self.ax_3d = self.fig_3d.gca(projection='3d')
            self.ax_3d.auto_scale_xyz([-3.75,3.75], [-4,4], [-3,3])
            self.ax_3d.view_init(azim=30)
            self.ax_3d.axis('off')
            plt.close()
    #}

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

                if mask is not None and not mask[r,c]: 
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

    def create_masks(self, flet_idx):
    #{
        if flet_idx is None: return {side: np.ones((3,3)) for side in SIDES}
         
        masks = {side: np.zeros((3,3)) for side in SIDES}
        for ci in self.viewable_cube.get_mask_cis(flet_idx):
            masks[ci[0]][ci[1]//3, ci[1]%3] = 1
        
        return masks
    #}

    def create_patches(self, flet_idx=None, seqnumb=0):
    #{
        rects = []
        masks = self.create_masks(flet_idx)     

        for side in SIDES:
            anchor = (self.ANCHOR_POS[side][0] + (self.OFFSET * seqnumb), self.ANCHOR_POS[side][1])
            rects.extend(self.get_plot_rects(np.reshape(self.viewable_cube.get_colors(side), (3,3)), anchor, masks[side]))
        
        return rects
    #}
    
    # Displays a single cube projection
    def draw_projection(self, flet_idx=None):
    #{
        fig = plt.figure(figsize=[4, 3])
        ax = fig.add_axes([0, 0, 1, 1])
        
        rects = self.create_patches(flet_idx)
        for r in rects: ax.add_patch(r)
        for ln in self.get_gridlines(): ax.add_line(ln)
 
        ax.axis('scaled')
        ax.axis('off')
        plt.show()
    #}
    
    def push_snapshot(self, flet_idx=None, caption=""):
    #{
        # Adjust to 5 cube images per row
        top = len(self.patch_sequence)-1
        if len(self.patch_sequence[top]) > 4:
            self.caption_sequence.append([])
            self.patch_sequence.append([])
            top = len(self.patch_sequence)-1

        seqnumb = len(self.patch_sequence[top])
        rects = self.create_patches(flet_idx, seqnumb)
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

    def draw_3d(self):
    #{
        # Get global index/ordering for facelet colors
        gpos = self.viewable_cube.facelet_matrix[2:] 
        if self.sm_adapter is not None: gpos = self.sm_adapter.get_global_positions()
        cindex = [np.nonzero(sum(gpos == np.broadcast_to(hpos, (54, 3)).T) == 3)[0][0]
                  for hpos in VectorCube._facelet_matrix[2:].T]

        ax3d = plt.figure().gca(projection='3d')
        for color, verts in zip(VectorCube._facelet_matrix[0, cindex], CubeView.get_poly_verts()):
            fpoly = Poly3DCollection([verts])
            fpoly.set_color(color_plot_map[color])
            fpoly.set_edgecolor('k')
            ax3d.add_collection3d(fpoly)

        ax3d.auto_scale_xyz([-3.75,3.75], [-4,4], [-3,3])
        ax3d.view_init(azim=30)
        ax3d.axis('off')
        plt.show()
    #}

    # Received move-by-move during recording. Note:
    # should be triggered BEFORE move executed on cube
    def sm_move_notification(self, rot_mats, mindex, dividx):
    #{
        # Create sequence of poly_verts for each step of the move
        # and do stepped rotations on appropriate (mindex) vertices
        move_verts = []
        for rmat in rot_mats:
            step_verts = CubeView.get_poly_verts().copy().astype(float)
            step_verts[mindex] = [np.matmul(rmat, verts.T).T for verts in step_verts[mindex]]
            move_verts.append(step_verts)

        # # Diddo for the dividers
        # move_dverts = []
        # for rmat in rot_mats:
        #     div_verts = CubeView.get_divider_verts().copy().astype(float)
        #     div_verts[dividx] = [np.matmul(rmat, verts.T).T for verts in div_verts[dividx]]
        #     move_dverts.append(div_verts)

        # Get global index/ordering for facelet colors
        gpos = self.sm_adapter.get_global_positions()
        cindex = [np.nonzero(sum(gpos == np.broadcast_to(hpos, (54, 3)).T) == 3)[0][0]
                  for hpos in VectorCube._facelet_matrix[2:].T]

        # Store the zipped color ordering and step verticies for animation run
        self.anim_zip_stack.extend([zip(VectorCube._facelet_matrix[0, cindex], 
                                    step_verts) for step_verts in move_verts])
        
        # # Diddo store on divider stack
        # self.divider_stack.extend(move_dverts)
    #}

    # Used by matplotlib.animation.FuncAnimation below,
    # NOT guaranteed to be called only once per run
    def init_animation(self):
    #{
        if len(self.fpoly_stack) == 0:
        #{
            # Create and init the 54 Poly3DCollection objects
            for color, verts in self.anim_zip_stack[0]:
                fpoly = Poly3DCollection([verts])
                fpoly.set_color(color_plot_map[color])
                fpoly.set_edgecolor('k')
                self.ax_3d.add_collection3d(fpoly)
                self.fpoly_stack.append(fpoly)

            # # Add black inner/divider polygons
            # for dverts in self.divider_stack[0]:
            #     fpoly = Poly3DCollection([dverts])
            #     fpoly.set_color('k')
            #     self.ax_3d.add_collection3d(fpoly)
            #     self.fpoly_stack.append(fpoly)
        #}

        return self.fpoly_stack
    #}
    
    # Used by matplotlib.animation.FuncAnimation below
    def get_next_frame(self, frame):
    #{
        # Update Poly3DCollection w/verts and color orderings per frame
        for i, (color, verts) in enumerate(self.anim_zip_stack[frame]):
            self.fpoly_stack[i].set_verts([verts])
            self.fpoly_stack[i].set_color(color_plot_map[color])
            self.fpoly_stack[i].set_edgecolor('k')
        
        # for j, dverts in enumerate(self.divider_stack[frame]):
        #     self.fpoly_stack[j+54].set_verts([dverts])
        #     self.fpoly_stack[j+54].set_color('k')
        #     self.fpoly_stack[j+54].set_edgecolor('k')

        return self.fpoly_stack
    #}

    def get_animation_3d(self):
        assert self.fig_3d is not None, "ERROR: called get_animation_3d() before recording moves"
        return FuncAnimation(self.fig_3d, self.get_next_frame, init_func=self.init_animation,
                             frames=len(self.anim_zip_stack), blit=True, repeat=False)
#}
