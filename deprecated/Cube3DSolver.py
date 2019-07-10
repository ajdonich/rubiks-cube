class Cube3DSolver:
#{
    def __init__(self):
    #{
        # Hyper-params
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
    #}

    def create_cube_model(self, mcontrollers=None, summarize=True):
    #{
        if mcontrollers is not None: self.clet_ctrls = mcontrollers
        for mctrl in self.clet_ctrls: mctrl.freeze_model()
        
        # Using cubelet models up to hidden_layer2 in cube model
        clet_instates = [ctrl.instate for ctrl in self.clet_ctrls]
        clet_hvals2 = keras.layers.concatenate([ctrl.hvals2 for ctrl in self.clet_ctrls])
        
        # Connect the full-cube-focused/tail-end of model
        x       = Dense(units=240, activation='relu')(clet_hvals2)
        x       = Dense(units=240, activation='relu')(x)
        qfinals = Dense(units=18, activation='linear')(x)
                
        # Construct and compile the full-cube model
        cube_model = Model(inputs=clet_instates, outputs=qfinals)
        cube_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        
        if summarize: cube_model.summary()        
        return cube_model
    #}
    
    def nn_state(self, cube):
        return [cubelet.state().reshape((1,12)) for cubelet in cube.cubelets]
    
    def step_cube(self, action, cube):
        cube.rotate(*CubeletSolver.ACTIONSET[action])  
        return cube.numb_solved(), cube.is_solved()
    
    def cube_policy(self, model, state, training=False):
    #{
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0,len(CubeletSolver.ACTIONSET))
        
        return np.argmax(model.predict(state, batch_size=1))
    #}
    
    # Plays 20 moves/rotations to solve for Rubik's cube
    def solve_cube(self, model, cube, nmoves=20, training=False):
    #{
        gamememory = []
        for mvnum in range(nmoves):
        #{
            # Initial state
            state  = self.nn_state(cube)
            action = self.cube_policy(model, state, training)

            # Update state and store training info
            reward, solved = self.step_cube(action, cube)
            gamememory.append((state, action, reward, self.nn_state(cube), solved))
            if solved: break
        #}
        
        return gamememory
    #}
    
    def train_cube_model(self, model, playmemory, batch_size=128):
    #{
        if len(playmemory) < batch_size: batch_size = len(playmemory)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)

        in_states = [np.zeros((batch_size, 12)) for i in range(Cube3D.NCUBELETS)]
        y_truish  = np.zeros((batch_size, len(CubeletSolver.ACTIONSET)))
        Qs_solved = np.zeros((batch_size, len(CubeletSolver.ACTIONSET)))
        minibatch = random.sample(playmemory, batch_size)
        
        for m in range(batch_size):
        #{
            # Unpack play memory
            state, action, reward, state_p, solved = minibatch[m]
        
            # Batch of input states 
            for i in range(Cube3D.NCUBELETS): in_states[i][m] = state[i]
            
            # Output target batch (r + discounted Rt+1)
            Qs_p = Qs_solved if solved else model.predict(state_p, batch_size=1)
            y_truish[m] = model.predict(state, batch_size=1)
            y_truish[m, action] = reward + (self.gamma * np.amax(Qs_p))            
        #}
                
        return model.fit(in_states, y_truish, batch_size=batch_size, epochs=1, verbose=0)
    #}
    
    def learn_cube(self, cube_model, episodes=10000):
    #{
        learning_cube = Cube3D()
        learnmemory, history = [], []
        
        self.epsilon = 1.0
        for j in range(episodes+1):
        #{
            # Reset env and train
            learning_cube.scramble()                
            learnmemory.extend(self.solve_cube(cube_model, learning_cube, training=True))
            history.append(self.train_cube_model(cube_model, learnmemory))
            if (j % 2000) == 0: print(f"  Training full Rubik's Cube: episode {j} of {episodes}")
        #}
        
        return history
    #}
#}