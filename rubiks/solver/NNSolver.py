import numpy as np
import keras, random, time
from keras.models import Model
from keras.layers import Dense, Input

from rubiks.model.CfopCube import CfopCube
from rubiks.model.DirectCube import DirectCube
from rubiks.model.VectorCube import VectorCube, GODS_NUMBER, SIDES, WHITE_CB, ORANGE_CB, GREEN_CB, RED_CB, BLUE_CB, YELLOW_CB

from rubiks.solver.DirectSolver import DirectSolver
from rubiks.solver.FaceletSolver import FaceletSolver


class NNSolver:
#{
    REC_INDEX = {
        WHITE_CB:  [[0,1,2,5,8,7,6,3,0],          [1,3,5,7],     4],
        ORANGE_CB: [[9,10,11,14,17,16,15,12,9],   [10,12,14,16], 13],
        GREEN_CB:  [[18,19,20,23,26,25,24,21,18], [19,21,23,25], 22],
        RED_CB:    [[27,28,29,32,35,34,33,30,27], [28,30,32,34], 31],
        BLUE_CB:   [[36,37,38,41,44,43,42,39,36], [37,39,41,43], 40],
        YELLOW_CB: [[45,46,47,50,53,52,51,48,45], [46,48,50,52], 49]
    }

    DIFF_SQ_OH_INDEX = {
        (4,9,9):   0,  (9,4,9):   1,  (9,9,4):   2,  (4,0,0):   3,  (0,4,0):   4,  (0,0,4):   5,
        (4,1,1):   6,  (1,4,1):   7,  (1,1,4):   8,  (4,25,25): 9,  (25,4,25): 10, (25,25,4): 11, 
        (25,4,1):  12, (25,1,4):  13, (4,25,1):  14, (1,25,4):  15, (4,1,25):  16, (1,4,25):  17,
        (9,1,0):   18, (9,0,1):   19, (1,9,0):   20, (0,9,1):   21, (1,0,9):   22, (0,1,9):   23,
        (25,9,0):  24, (25,0,9):  25, (9,25,0):  26, (0,25,9):  27, (9,0,25):  28, (0,9,25):  29,
        (36,4,0):  30, (36,0,4):  31, (4,36,0):  32, (0,36,4):  33, (4,0,36):  34, (0,4,36):  35,
        (36,16,4): 36, (36,4,16): 37, (16,36,4): 38, (4,36,16): 39, (16,4,36): 40, (4,16,36): 41,
        (16,4,0):  42, (16,0,4):  43, (4,16,0):  44, (0,16,4):  45, (4,0,16):  46, (0,4,16):  47,
        (16,9,1):  48, (16,1,9):  49, (9,16,1):  50, (1,16,9):  51, (9,1,16):  52, (1,9,16):  53,
        (25,16,9): 54, (25,9,16): 55, (16,25,9): 56, (9,25,16): 57, (16,9,25): 58, (9,16,25): 59
    }

    def __init__(self):
        self.gamma = 0.95
        self.early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # Pretty straightforward dense network
    def create_policy_model(self, input_dim=72, summarize=True):
    #{
        # Prime input, connect layers, get ref to output
        X_input  = Input(shape=(input_dim,))
        X        = Dense(units=1024, activation='relu')(X_input)
        X        = Dense(units=512, activation='relu')(X)
        X_policy = Dense(units=256, activation='relu')(X)
        P_output = Dense(18, activation='softmax', name='P_output')(X_policy)
                
        # Construct and compile the full-cube model
        cube_model = Model(inputs=X_input, outputs= P_output)
        
        adam_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)       
        cube_model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        if summarize: cube_model.summary()        
        return cube_model
    #}

    def create_dot_input(self, cube):
    #{
        dot_state = []
        for side in SIDES:
            ring = NNSolver.REC_INDEX[side][0]
            edges = NNSolver.REC_INDEX[side][1]
            dot_state.extend([cube.dot((ring[i], ring[i+1])) for i in range(len(ring)-1)])
            dot_state.extend([cube.dot((eg, NNSolver.REC_INDEX[side][2])) for eg in edges])
        
        return np.array(dot_state)
    #}

    def create_diffsq_input(self, cube):
    #{
        one_hot_cube = np.zeros((72,60), dtype=int)
        diff_sq = (cube.facelet_matrix[2:, VectorCube._order_cn_cnt] - cube.facelet_matrix[2:, VectorCube._order_ed])**2
        for i, diff in enumerate(diff_sq.T): one_hot_cube[i, NNSolver.DIFF_SQ_OH_INDEX[tuple(diff)]] = 1
        return one_hot_cube.flatten()
    #}

    def create_dance_input(self, direct_cube):
        return np.concatenate(DirectSolver.dance_heuristics(direct_cube, value=False)).astype(int)

    def generate_dataset(self, create_fcn, sz_factor=10000, validation_split=0.2):
    #{
        train_sz = int(sz_factor * (1.0 - validation_split))
        Xl, Yl, Xlv, Ylv = [], [], [], []

        # Training set
        for i in range(train_sz):
        #{
            cube = DirectCube()
            moves, invmoves = cube.trace_scramble()
            for mv in moves: Xl.append(create_fcn(cube.rotate(mv)))
            for invmv in reversed(invmoves): Yl.append(CfopCube.ACTIONS[invmv])
        #}

        # Validation set
        for i in range(train_sz, sz_factor):
        #{
            cube = DirectCube()
            moves, invmoves = cube.trace_scramble()
            for mv in moves: Xlv.append(create_fcn(cube.rotate(mv)))
            for invmv in reversed(invmoves): Ylv.append(CfopCube.ACTIONS[invmv])
        #}

        # Prepare final NN input data blocks
        X, Xval = np.array(Xl), np.array(Xlv)
        Yoh = np.zeros((len(Yl), len(VectorCube.MOVES)), dtype=int)
        Yohval = np.zeros((len(Ylv), len(VectorCube.MOVES)), dtype=int)

        # Convert targets/outputs to one-hot format
        for i, action in enumerate(Yl): Yoh[i, action] = 1
        for i, action in enumerate(Ylv): Yohval[i, action] = 1

        print((f"Generated dataset:\n X.shape = {X.shape}\n Yoh.shape = {Yoh.shape}\n"
               f" Xval.shape = {Xval.shape}\n Yohval.shape = {Yohval.shape}"))

        return X, Yoh, Xval, Yohval
    #}

    # def create_value_policy_model(self, summarize=True):
    # #{
    #     # Prime input, connect layers, get ref to output
    #     X_input  = Input(shape=(144,))
    #     X        = Dense(units=4096, activation='relu')(X_input)
    #     X        = Dense(units=2048, activation='relu')(X)
        
    #     X_value  = Dense(units=512, activation='relu')(X)
    #     Q_output = Dense(1, activation='linear', name='Q_output')(X_value)
        
    #     X_policy = Dense(units=512, activation='relu')(X)
    #     P_output = Dense(1, activation='softmax', name='P_output')(X_policy)
                
    #     # Construct and compile the full-cube model
    #     cube_model = Model(inputs=X_input, outputs=[Q_output, P_output])
        
    #     opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
    #     early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    #     # cube_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        
    #     cube_model.compile(optimizer=opt, metrics=['mae', 'categorical_accuracy'],
    #         loss={'Q_output': 'mean_squared_error', 'P_output': 'categorical_crossentropy'})

    #     if summarize: cube_model.summary()        
    #     return cube_model
    # #}

#     def generate_dataset(self, sz_factor=300):
#     #{
#         invact = [1,-1,0]
#         Xl, Yl, Xlv, Ylv = [], [], [], []
#         for gn in range(1, GODS_NUMBER+1):
#         #{
#             for iters in range(int(gn * sz_factor * (np.log(gn)+1))):
#             #{
#                 cube = Opticube()
#                 if np.random.random() < 0.2:
#                     for action in np.random.randint(len(Opticube.MOVES), size=gn):
#                         Xlv.append(cube.rotate(Opticube.MOVES[action]).state())
#                         Ylv.append(np.zeros((len(Opticube.MOVES),)))
#                         Ylv[-1][action + invact[action%3]] = 1.0
#                 else:
#                     for action in np.random.randint(len(Opticube.MOVES), size=gn):
#                         Xl.append(cube.rotate(Opticube.MOVES[action]).state())
#                         Yl.append(np.zeros((len(Opticube.MOVES),)))
#                         Yl[-1][action + invact[action%3]] = 1.0
#             #}
#         #}
        
#         X, Y, Xval, Yval = np.array(Xl), np.array(Yl), np.array(Xlv), np.array(Ylv)
#         print((f"Generated dataset:\n X.shape = {X.shape}\n Y.shape = {Y.shape}\n"
#                f" Xval.shape = {Xval.shape}\n Yval.shape = {Yval.shape}"))
#         return X, Y, Xval, Yval
#     #}
    
#     # Plays max of 128 moves/rotations trying to solve the cube
#     def solve_cube(self, model, cube, nmoves=128, training=False):
#     #{        
#         gamememory = []
#         UCTNode.reset_tree()
#         rootnode = UCTNode(cube.distance())
        
#         for mvnum in range(nmoves):
#         #{
#             # Initial state
#             state  = self.nn_state(cube)
            
#             # Determine and apply the best next move
#             vnode = self.uc_tree_search(model, cube, rootnode)
#             solved = cube.rotate(Opticube.MOVES[vnode.action]).is_solved()

#             # Store training information
#             gamememory.append((state, vnode.action, vnode.avg_value, self.nn_state(cube), solved))
            
#             if solved: break
#             else: rootnode = vnode.set_root()
#         #}
        
#         return gamememory
#     #}
    
#     def train_cube_model(self, model, playmemory, batch_size=128):
#     #{
#         if len(playmemory) < batch_size: batch_size = len(playmemory)

#         in_states = np.zeros((batch_size, 162))
#         y_truish  = np.zeros((batch_size, len(Opticube.MOVES)))
#         Qs_solved = np.zeros((batch_size, len(Opticube.MOVES)))
#         minibatch = random.sample(playmemory, batch_size)
        
#         for m in range(batch_size):
#         #{
#             # Unpack play memory
#             state, action, reward, state_p, solved = minibatch[m]
        
#             # Batch of input states 
#             in_states[m] = state
            
#             # Output target batch (r + discounted Rt+1)
#             Qs_p = Qs_solved if solved else model.predict(state_p, batch_size=1)
#             y_truish[m] = model.predict(state, batch_size=1)
#             y_truish[m, action] = reward + (self.gamma * np.amax(Qs_p))            
#         #}
                
#         return model.fit(in_states, y_truish, batch_size=batch_size, epochs=1, verbose=0)
#     #}
    
#     def learn_cube(self, cube_model, episodes=10): #episodes=450
#     #{
#         learning_cube = Opticube()
#         learnmemory, history = [], []
#         initial_times, final_times, tree_stats = [],[],[]
        
#         for j in range(episodes+1):
#         #{
#             initial_times.append(time.time())
        
#             # Reset env and train
#             learning_cube.scramble()                
#             learnmemory.extend(self.solve_cube(cube_model, learning_cube, training=True))
#             history.append(self.train_cube_model(cube_model, learnmemory))
            
#             final_times.append(time.time())
#             tree_stats.append(UCTNode.total_visits)
            
# #             if (j % 2000) == 0:
#             if (j % 2) == 0:
#                 tree = np.mean(tree_stats)
#                 duration = np.mean(np.array(final_times) - np.array(initial_times))
#                 print(f" Training Opticube: episode {j} of {episodes}")
#                 print(f"   Average episode duration: {duration} sec")
#                 print(f"   Average tree nodes: {tree[0]}, visits: {tree[1]}")
#                 initial_times, final_times, tree_stats = [],[],[]
#         #}
        
#         return history
#     #}
    
#     def uc_tree_search(self, model, opticube, rootnode, iterations=1000):
#     #{
#         # UCT loop (w/no MC rollout)
#         for i in range(iterations):
#         #{            
#             node = rootnode
#             cube = Opticube(opticube)
            
#             # Select (down to leaf)
#             while node.child_nodes:
#                 node = node.select_uct_child()
#                 cube.rotate(Opticube.MOVES[node.action])
                
#             # Expand fully and evaluate via NN
#             for action in range(len(Opticube.MOVES)): node.add_child(action)
#             action = np.argmax(model.predict(self.nn_state(cube), batch_size=1))
#             dist = cube.rotate(Opticube.MOVES[action]).distance()
#             node = node.child_nodes[action]
            
#             # Backup
#             while node:
#                 node.update(dist)
#                 node = node.parent_node
#         #}
        
#         return rootnode.select_fav_child()
#     #}
    
#}