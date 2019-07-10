import keras
import random 

from keras.models import Model
from keras.layers import Dense, Input

class CubeletSolver:
#{
    ACTIONSET = [(sd,ang) for sd in SIDES for ang in [90,-90,180]]

    class CletModelCtrl:
    #{
        def __init__(self, clet_i, summarize=False):
        #{
            self.clet_i = clet_i
            self.history = []
            
            # Construct trainable layers
            self.xhidden_layer1 = Dense(units=12, activation='relu')
            self.xhidden_layer2 = Dense(units=12, activation='relu')
            self.xoutput_layer  = Dense(units=18, activation='linear')

            # Prime input, connect layers, get ref to output
            self.instate = Input(shape=(12,))
            self.hvals1  = self.xhidden_layer1(self.instate)
            self.hvals2  = self.xhidden_layer2(self.hvals1)
            self.qvals   = self.xoutput_layer(self.hvals2)
                        
            # Construct and compile the cubelet model
            self.model = Model(inputs=self.instate, outputs=self.qvals)
            self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
            if summarize: self.model.summary()
        #}
        
        def freeze_model(self, freeze=True):
        #{
            self.xhidden_layer1.trainable = not freeze
            self.xhidden_layer2.trainable = not freeze
            self.xoutput_layer.trainable  = not freeze
            self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        #}
        
        def save_model(self):
            tri_name = tri_color_name(Cube3D.CLET_KEYS[self.clet_i])
            self.model.save_weights(f'clet_models/clet_{tri_name}_weights.h5')
        
        def load_model(self):
            tri_name = tri_color_name(Cube3D.CLET_KEYS[self.clet_i])
            self.model.load_weights(f'clet_models/clet_{tri_name}_weights.h5')
    #}

    def __init__(self):
    #{
        # Lazy initialize
        self.clet_ctrls = []
        
        # Hyper-params
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
    #}
    
    def nn_state(self, cubelet):
        return cubelet.state().reshape((1,12))
    
    def is_same_state(self, state_a, state_b):
        return sum(np.ndarray.flatten(state_a) == np.ndarray.flatten(state_b)) == 12

    def save_models(self, mcontrollers=None):
        if mcontrollers is not None: self.clet_ctrls = mcontrollers
        for mctrl in self.clet_ctrls: mctrl.save_model()
    
    def load_models(self):
    #{
        for i in range(len(Cube3D.CLET_KEYS)):
            mctrl = CubeletSolver.CletModelCtrl(i)
            self.clet_ctrls.append(mctrl)
            self.clet_ctrls[i].load_model()
        
        return self.clet_ctrls
    #}
    
    def get_model_ctrl(self, ctrl_id):
    #{
        # ctrl_id can be either tri-colors tuple or clet index
        if type(ctrl_id) is int: ctrl_id = Cube3D.CLET_KEYS[ctrl_id]
        return self.clet_ctrls.get(ctrl_id, None)
    #}
    
    def step_cubelet(self, action, cubelet):
    #{
        # Test if initial state already solved
        had_been_solved = Cube3D.clet_is_solved(cubelet)
        
        # Update cubelet state relative to action
        side, angle = CubeletSolver.ACTIONSET[action]
        if cubelet.ison(side): cubelet.apply_rotation\
            (Cube3D.rotation((side, angle)))
                
        # Calc reward
        if Cube3D.clet_is_solved(cubelet):
            if not had_been_solved: reward = 1.5
            else:                   reward = 0.5
        elif had_been_solved:       reward = -0.5
        else:                       reward = -0.5

        return reward
    #}
    
    def clet_policy(self, model, state, training=False):
    #{
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0,len(CubeletSolver.ACTIONSET))
        
        return np.argmax(model.predict(state, batch_size=1))
    #}
    
    # Plays 20 moves/rotations to solve for cubelet; cubelet.copy()
    # used to isolate cubelet-only moves from confusing a client cube 
    def solve_cubelet(self, model, cubelet, nmoves=20, training=False):
    #{
        gamememory = []
        cubelet = cubelet.copy()
        for mvnum in range(nmoves):
        #{
            # Initial state
            state  = self.nn_state(cubelet)
            action = self.clet_policy(model, state, training)

            # Update state and store training info
            reward = self.step_cubelet(action, cubelet)
            gamememory.append((state, action, reward, self.nn_state(cubelet)))
        #}
        
        return gamememory
    #}
    
    def train_cubelet_model(self, model, playmemory, batch_size=128):
    #{
        if len(playmemory) < batch_size: batch_size = len(playmemory)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)
        
        in_state = np.zeros((batch_size, 12))
        y_truish = np.zeros((batch_size, len(CubeletSolver.ACTIONSET)))
        minibatch = random.sample(playmemory, batch_size)
        
        for m in range(batch_size):
        #{
            # Unpack play memory
            state, action, reward, state_p = minibatch[m]
        
            # Input state batch
            in_state[m] = state
            
            # Output target batch (r + discounted Rt+1)
            Qs_p = model.predict(state_p, batch_size=1)
            y_truish[m] = model.predict(state, batch_size=1)
            y_truish[m, action] = reward + (self.gamma * np.amax(Qs_p))            
        #}
                
        return model.fit(in_state, y_truish, batch_size=batch_size, epochs=1, verbose=0)
    #}
    
    def learn_cubelet(self, clet_i, episodes=10000):
    #{
        learnmemory = []
        learning_cube = Cube3D()
        
        # Initialize a model controller
        cubelet = learning_cube.cubelets[clet_i]
        mctrl = CubeletSolver.CletModelCtrl(clet_i, summarize=True)
        
        self.epsilon = 1.0
        for j in range(episodes+1):
        #{
            # Reset env and train
            learning_cube.scramble()                
            learnmemory.extend(self.solve_cubelet(mctrl.model, cubelet, training=True))
            mctrl.history.append(self.train_cubelet_model(mctrl.model, learnmemory, batch_size=32))
            if (j % 2000) == 0: print(f"  Training {cubelet.name()}: episode {j} of {episodes}")
        #}
        
        return mctrl
    #}
#}