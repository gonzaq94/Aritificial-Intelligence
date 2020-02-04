### ELU 616 
### Author: Carlos Lassance, Nicolas Farrugia
### Lab Session on Reinforcement learning
### The goal of this short lab session is to quickly put in practice a simple way to use reinforcement learning to train an agent to play PyRat
### You will be using Experience Replay: while playing, the agent will 'remember' the moves he performs, and will subsequently train itself to predict what should be the next move, depending on how much reward is associated with the moves.

### Usage : python main.py
### Change the parameters directly in this file. 

### GOAL : complete this file (main.py) in order to perform the full training and testing procedure using reinforcement learning and experience replay.

### When training is finished, copy both the AIs/numpy_rl_reload.py file and the save_rl folder into your pyrat folder, and run a pyrat game with the appropriate parameters using the numpy_rl_reload.py as AI

import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from AIs import manh, numpy_rl_reload
import matplotlib.pyplot as plt

### The game.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent
import game

### The rl.py file describes the reinforcement learning procedure, including Q-learning, Experience replay, and Stochastic Gradient Descent (SGD). SGD is used to approximate the Q-function
import rl_non_linear
loss_ = []


### This set of parameters can be changed in your experiments.
### Definitions :
### - An iteration of training is called an Epoch. It correspond to a full play of a PyRat game. 
### - an experience is a set of  vectors < s, a, r, s’ > describing the consequence of being in state s, doing action a, receiving reward r, and ending up in state s'. See in file rl.py to see how the experience replay buffer is implemented. 
### - a batch is a set of experiences we use for training during one epoch. We draw batches from the experience replay buffer.


epoch = 3000 ### Total number of epochs that will be done

max_memory = 1000 # Maximum number of experiences we are storing
number_of_batches = 8 # Number of batches per epoch
batch_size = 32 # Number of experiences we use for training per batch
width = 21 # Size of the playing field
height = 15 # Size of the playing field
cheeses = 40 # number of cheeses in the game
opponent = manh # AI used for the opponent

### If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters
load = False
save = True


env = game.PyRat()
exp_replay = rl_non_linear.ExperienceReplay(max_memory=max_memory)
name = 'model_linear'
model = rl_non_linear.NNModel(env.observe()[0],name)

if load:
    model.load()

def play(model,epochs,train=True):

    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    loss_hist = []
    cheeses = []
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0

    for e in tqdm(range(epochs)):
        env.reset() #init parameters
        game_over = False

        # Get the current state of the environment

        # return current canvas
        current_state = env.observe()

        # Play a full game until game is over
        while not game_over:
 
            ### Predict the Q value for the current state
            Qs = model.predict([current_state])[0]
            
            ### Pick the next action that maximizes the Q value
            action = np.argmax(Qs)                       
            
            # apply action, get rewards and new statelose_cnt0
            new_state, reward, game_over = env.act(action)
                       
            if game_over: # Statistics
                steps += env.round # Statistics
                if env.score > env.enemy_score: # Statistics
                    win_cnt += 1 # Statistics
                elif env.score == env.enemy_score: # Statistics
                    draw_cnt += 1 # Statistics
                else: # Statistics
                    lose_cnt += 1 # Statistics
                cheese = env.score # Statistics


            ## Create an experience array using previous state, the action performed, the reward obtained and the new state. The vector has to be in this order.
            ## Store in the experience replay buffer an the experience and end game
            exp = [current_state,action,reward,new_state]
            exp_replay.remember(exp,game_over)
            current_state = new_state
         
            
        win_hist.append(win_cnt) # Statistics
        loss_hist.append(lose_cnt) # Statistics
        cheeses.append(cheese) # Statistics

        if train:

            # Train using experience replay. For each batch, get a set of experiences (state, action, new state) that were stored in the buffer. Use this batch to train the model.
            for i in range(0,number_of_batches):
                inputs, Q = exp_replay.get_batch(model,batch_size) #batch 10
                loss = model.train_on_batch(inputs,Q)
                loss_.append(loss)
        else:
            loss = 0

        if (e+1) % 100 == 0: # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            string = "Epoch {:03d}/{:03d} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}, Loss {}".format(
                        e,epochs, cheese_np.sum(), 
                        cheese_np[-100:].sum(),win_cnt,draw_cnt,lose_cnt, 
                        win_cnt-last_W,draw_cnt-last_D,lose_cnt-last_L,steps/100,loss)
            print(string)
        
            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt                

print("Training")
play(model,epoch,True)
if save:
    model.save()
print("Training done")
print("Testing")
play(model,epoch,False)
print("Testing done")
np.save('win_hist'+name+'.npy',win_hist)
np.save('lose_hist'+name+'.npy',losss_hist)
