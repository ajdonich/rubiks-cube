import math
import numpy as np

from rubiks.model.VectorCube import VectorCube, SIDES, WHITE_CB, W, ORANGE_CB, O, GREEN_CB, G, RED_CB, R, BLUE_CB, B, YELLOW_CB, Y

# CFOP Algorithm Stages
CROSS = 1       # first layer cross/edge-cubelets
F2L   = 2       # first two layers
OLL   = 3       # orient last layer
PLL   = 4       # permute last layer
FULL  = 5       # extra: full cube solution

cfop_map = { 0:'N/A', CROSS:'CROSS', F2L:'F2L', OLL:'OLL', PLL:'PLL', FULL:'FULL' }

def cfop_name(state): return cfop_map[state]

class CfopCube(VectorCube):
#{
    CFOP_IDXS = {
        (CROSS, WHITE_CB):  np.array([1,3,5,7]),
        (CROSS, ORANGE_CB): np.array([10,12,14,16]),
        (CROSS, GREEN_CB):  np.array([19,21,23,25]),
        (CROSS, RED_CB):    np.array([28,30,32,34]),
        (CROSS, BLUE_CB):   np.array([37,39,41,43]),
        (CROSS, YELLOW_CB): np.array([46,48,50,52]),

        # In each quad, quad[0], quad[2] is the corner cubelet, quad[1], quad[3] is the edge
        (F2L, WHITE_CB):  np.array([(38,41,9,12),(36,39,29,32),(18,21,11,14),(20,23,27,30)]),
        (F2L, ORANGE_CB): np.array([(38,37,0,1),(18,19,6,7),(44,43,51,52),(24,25,45,46)]),
        (F2L, GREEN_CB):  np.array([(6,3,11,10),(8,5,27,28),(45,48,17,16),(47,50,33,34)]),
        (F2L, RED_CB):    np.array([(20,19,8,7),(36,37,2,1),(26,25,47,46),(42,43,53,52)]),
        (F2L, BLUE_CB):   np.array([(2,5,29,28),(0,3,9,10),(53,50,35,34),(51,48,15,16)]),
        (F2L, YELLOW_CB): np.array([(24,21,17,14),(26,23,33,30),(44,41,15,12),(42,39,35,32)]),

        (OLL, WHITE_CB):  np.array([45,46,47,48,50,51,52,53]),
        (OLL, ORANGE_CB): np.array([27,28,29,30,32,33,34,35]),
        (OLL, GREEN_CB):  np.array([36,37,38,39,41,42,43,44]),
        (OLL, RED_CB):    np.array([9,10,11,12,14,15,16,17]),
        (OLL, BLUE_CB):   np.array([18,19,20,21,23,24,25,26]),
        (OLL, YELLOW_CB): np.array([0,1,2,3,5,6,7,8])
    }

    CUBLET_PAIRS = {
        WHITE_CB:  [(0,9),(0,38),(1,37),(2,29),(2,36),(5,28),(8,20),(8,27),(7,19),(6,11),(6,18),(3,10)],
        ORANGE_CB: [(9,0),(9,38),(10,3),(11,6),(11,18),(14,21),(17,24),(17,45),(16,48),(15,44),(15,51),(12,41)],
        GREEN_CB:  [(18,6),(18,11),(19,7),(20,8),(20,27),(23,30),(26,33),(26,47),(25,46),(24,17),(24,45),(21,14)],
        RED_CB:    [(27,8),(27,20),(28,5),(29,2),(29,36),(32,39),(35,42),(35,53),(34,50),(33,26),(33,47),(30,23)],
        BLUE_CB:   [(36,2),(36,29),(37,1),(38,0),(38,9),(41,12),(44,15),(44,51),(43,52),(42,35),(42,53),(39,32)],
        YELLOW_CB: [(45,17),(45,24),(46,25),(47,26),(47,33),(50,34),(53,35),(53,42),(52,43),(51,15),(51,44),(48,16)]
    }

    # Begin CfopCube implementation
    def __init__(self, copycube=None):
        super(CfopCube, self).__init__(copycube)
#}