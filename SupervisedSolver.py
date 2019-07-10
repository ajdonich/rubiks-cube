import keras, random, time
from keras.models import Model
from keras.layers import Dense, Input

from Opticube import Opticube

class SupervisedSolver:
#{
    def __init__(self):
        self.gamma = 0.95

    def create_supervised_model(self, summarize=True):
    #{
        # Prime input, connect layers, get ref to output
        X_input  = Input(shape=(144,))
        X        = Dense(units=4096, activation='relu')(X_input)
        X        = Dense(units=2048, activation='relu')(X)
        
        X_value  = Dense(units=512, activation='relu')(X)
        Q_output = Dense(1, activation='linear', name='Q_output')(X_value)
        
        X_policy = Dense(units=512, activation='relu')(X)
        P_output = Dense(1, activation='softmax', name='P_output')(X_policy)
                
        # Construct and compile the full-cube model
        cube_model = Model(inputs=X_input, outputs=[Q_output, P_output])
        
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
        early_stop = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
        # cube_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        
        cube_model.compile(optimizer=opt, metrics=['mae', 'categorical_accuracy'],
            loss={'Q_output': 'mean_squared_error', 'P_output': 'categorical_crossentropy'})

        if summarize: cube_model.summary()        
        return cube_model
    #}

    def generate_dataset(self, sz_factor=300):
    #{
        invact = [1,-1,0]
        Xl, Yl, Xlv, Ylv = [], [], [], []
        for gn in range(1, GODS_NUMBER+1):
        #{
            for iters in range(int(gn * sz_factor * (np.log(gn)+1))):
            #{
                cube = Opticube()
                if np.random.random() < 0.2:
                    for action in np.random.randint(len(Opticube.MOVES), size=gn):
                        Xlv.append(cube.rotate(Opticube.MOVES[action]).state())
                        Ylv.append(np.zeros((len(Opticube.MOVES),)))
                        Ylv[-1][action + invact[action%3]] = 1.0
                else:
                    for action in np.random.randint(len(Opticube.MOVES), size=gn):
                        Xl.append(cube.rotate(Opticube.MOVES[action]).state())
                        Yl.append(np.zeros((len(Opticube.MOVES),)))
                        Yl[-1][action + invact[action%3]] = 1.0
            #}
        #}
        
        X, Y, Xval, Yval = np.array(Xl), np.array(Yl), np.array(Xlv), np.array(Ylv)
        print((f"Generated dataset:\n X.shape = {X.shape}\n Y.shape = {Y.shape}\n"
               f" Xval.shape = {Xval.shape}\n Yval.shape = {Yval.shape}"))
        return X, Y, Xval, Yval
    #}
    
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
    
#}