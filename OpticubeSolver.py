import keras, random, time
from keras.models import Model
from keras.layers import Dense, Input

class UCTNode:
#{
    UCTFACTOR = math.sqrt(2.0)
    
    total_visits = 0
    total_nodes = 0

    @classmethod
    def reset_tree(cls):    
        cls.total_visits = 0
        cls.total_nodes = 0
    
    def __init__(self, distance=2.25, action=None, parent=None):
    #{
        # Action that produced this node/state
        self.action = action
        
        self.numb_visits = 0
        self.child_nodes = []
        self.parent_node = parent

        # Value ends up in set (0, 1.25] 
        self.value = 2.25 - distance
        self.avg_value = self.value
        
        UCTNode.total_nodes += 1
    #}
    
    def __repr__(self):
        return (f"Action:  {self.action}\n"
                f"Visits:  {self.numb_visits}\n"
                f"Value:   {self.value}\n"
                f"Avg Val: {self.avg_value})")
    
    def set_root(self):
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
        return max(self.child_nodes, key=lambda cld: cld.avg_value + cld.ucbonus())
    
    def select_fav_child(self):
        return max(self.child_nodes, key=lambda cld: cld.numb_visits)
    
    def add_child(self, action):
    #{
        child = UCTNode(action=action, parent=self)
        self.child_nodes.append(child)
        return child
    #}
    
    def update(self, distance):
    #{
        self.numb_visits += 1
        self.value += (2.25 - distance)
        self.avg_value = self.value/self.numb_visits
        UCTNode.total_visits += 1
    #}
#}

class OpticubeSolver:
#{
    def __init__(self):
        self.gamma = 0.95

    def create_opticube_model(self, summarize=True):
    #{  
        # Prime input, connect layers, get ref to output
        X_input  = Input(shape=(162,))
        X        = Dense(units=162, activation='relu')(X_input)
        X        = Dense(units=162, activation='relu')(X)
        X        = Dense(units=162, activation='relu')(X)
        Q_output = Dense(units=18, activation='linear')(X)
                
        # Construct and compile the full-cube model
        cube_model = Model(inputs=X_input, outputs=Q_output)
        cube_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
        if summarize: cube_model.summary()        
        return cube_model
    #}
    
    def nn_state(self, cube):
        return cube.state().reshape((1, 162))
    
    # Plays max of 128 moves/rotations trying to solve the cube
    def solve_cube(self, model, cube, nmoves=128, training=False):
    #{        
        gamememory = []
        UCTNode.reset_tree()
        rootnode = UCTNode(cube.distance())
        
        for mvnum in range(nmoves):
        #{
            # Initial state
            state  = self.nn_state(cube)
            
            # Determine and apply the best next move
            vnode = self.uc_tree_search(model, cube, rootnode)
            solved = cube.rotate(Opticube.MOVES[vnode.action]).is_solved()

            # Store training information
            gamememory.append((state, vnode.action, vnode.avg_value, self.nn_state(cube), solved))
            
            if solved: break
            else: rootnode = vnode.set_root()
        #}
        
        return gamememory
    #}
    
    def train_cube_model(self, model, playmemory, batch_size=128):
    #{
        if len(playmemory) < batch_size: batch_size = len(playmemory)

        in_states = np.zeros((batch_size, 162))
        y_truish  = np.zeros((batch_size, len(Opticube.MOVES)))
        Qs_solved = np.zeros((batch_size, len(Opticube.MOVES)))
        minibatch = random.sample(playmemory, batch_size)
        
        for m in range(batch_size):
        #{
            # Unpack play memory
            state, action, reward, state_p, solved = minibatch[m]
        
            # Batch of input states 
            in_states[m] = state
            
            # Output target batch (r + discounted Rt+1)
            Qs_p = Qs_solved if solved else model.predict(state_p, batch_size=1)
            y_truish[m] = model.predict(state, batch_size=1)
            y_truish[m, action] = reward + (self.gamma * np.amax(Qs_p))            
        #}
                
        return model.fit(in_states, y_truish, batch_size=batch_size, epochs=1, verbose=0)
    #}
    
    def learn_cube(self, cube_model, episodes=10): #episodes=450
    #{
        learning_cube = Opticube()
        learnmemory, history = [], []
        initial_times, final_times, tree_stats = [],[],[]
        
        for j in range(episodes+1):
        #{
            initial_times.append(time.time())
        
            # Reset env and train
            learning_cube.scramble()                
            learnmemory.extend(self.solve_cube(cube_model, learning_cube, training=True))
            history.append(self.train_cube_model(cube_model, learnmemory))
            
            final_times.append(time.time())
            tree_stats.append(UCTNode.total_visits)
            
#             if (j % 2000) == 0:
            if (j % 2) == 0:
                tree = np.mean(tree_stats)
                duration = np.mean(np.array(final_times) - np.array(initial_times))
                print(f" Training Opticube: episode {j} of {episodes}")
                print(f"   Average episode duration: {duration} sec")
                print(f"   Average tree nodes: {tree[0]}, visits: {tree[1]}")
                initial_times, final_times, tree_stats = [],[],[]
        #}
        
        return history
    #}
    
    def uc_tree_search(self, model, opticube, rootnode, iterations=1000):
    #{
        # UCT loop (w/no MC rollout)
        for i in range(iterations):
        #{            
            node = rootnode
            cube = Opticube(opticube)
            
            # Select (down to leaf)
            while node.child_nodes:
                node = node.select_uct_child()
                cube.rotate(Opticube.MOVES[node.action])
                
            # Expand fully and evaluate via NN
            for action in range(len(Opticube.MOVES)): node.add_child(action)
            action = np.argmax(model.predict(self.nn_state(cube), batch_size=1))
            dist = cube.rotate(Opticube.MOVES[action]).distance()
            node = node.child_nodes[action]
            
            # Backup
            while node:
                node.update(dist)
                node = node.parent_node
        #}
        
        return rootnode.select_fav_child()
    #}
    
#     def uc_tree_search_log(self, model, opticube, rootnode, iterations=1000):
#     #{
#         start = time.time()
#         proto_avg = {'ctor': 0,
#                      'select': 0,
#                      'expand': 0,
#                      'backup': 0}
        
#         # UCT loop (w/no MC rollout)
#         for i in range(iterations):
#         #{
#             initial = time.time()
            
#             node = rootnode
#             cube = Opticube(opticube)
            
#             ctor = time.time()

#             # Select (down to leaf)
#             while node.child_nodes:
#                 node = node.select_uct_child()
#                 cube.rotate(Opticube.MOVES[node.action])
                
#             select = time.time()

#             # Expand fully and evaluate via NN
#             for action in range(len(Opticube.MOVES)): node.add_child(action)
#             action = np.argmax(model.predict(self.nn_state(cube), batch_size=1))
#             dist = cube.rotate(Opticube.MOVES[action]).distance()
#             node = node.child_nodes[action]
            
#             expand = time.time()

#             # Backup
#             while node:
#                 node.update(dist)
#                 node = node.parent_node
                
#             backup = time.time()
            
#             proto_avg['ctor'] += (ctor - initial)
#             proto_avg['select'] += (select - ctor)
#             proto_avg['expand'] += (expand - select)
#             proto_avg['backup'] += (backup - expand)
#         #}
        
#         vnode = rootnode.select_fav_child()
        
#         duration = time.time() - start
#         a = proto_avg['ctor']  / duration
#         b = proto_avg['select'] / duration
#         c = proto_avg['expand'] / duration
#         d = proto_avg['backup'] / duration
        
#         print(f"  {iterations} UCT search iterations: {time.time() - start} sec")
#         print(f"    ctor: {a}, select: {b}, expand: {c}, backup: {d}")
              
#         return vnode
#     #}
#}