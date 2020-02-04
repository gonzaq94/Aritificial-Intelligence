#### Playing Pyrat using a trained sklearn classifier
#### April 22nd 2019

###############################
# Team name to be displayed in the game 
TEAM_NAME = "testPlaying"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time
from sklearn.externals import joblib

### CORRECTED VERSION
### We directly add the code for the convert_input function here, so as to have all in a single file for the challenge (easier to manage for us ;) ) 
def generate_canvas_opponnent(mux,muy,sigma = 5,mazeWidth=21,mazeHeight=15): 
    x, y = np.meshgrid(np.linspace(-mazeWidth,mazeWidth,2*mazeWidth-1), np.linspace(-mazeHeight,mazeHeight,2*mazeHeight-1))
    return np.exp(-( ((x-mux)**2+(y-muy)**2) / ( 2.0 * sigma**2 ) ) )

def convert_input_2(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2*mazeHeight-1,2*mazeWidth-1,1)
    #29x41
    # We initialize a canvas with only zeros
    canvas = np.zeros(im_size)


    (x,y) = player
    (x_op,y_op) = opponent

    # fill in the first layer of the canvas with the value 1 at the location of the cheeses, relative to the position of the player (i.e. the canvas is centered on the player location)
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        x_ = y_cheese+center_y-y
        y_ = x_cheese+center_x-x
        canvas[(x_-2):(x_+3),(y_-2):(y_+3),0] += 0.1
        canvas[(x_-1):(x_+2),(y_-1):(y_+2),0] += 0.2
        canvas[x_,y_,0] =+1        
    return canvas


###############################
# Please put your global variables here

# Global variables
global model
    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model
    ### Here we load the previously trained model
    model = joblib.load('mlp_classifier_moves_smell.pkl') 

###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score

    # Transform the input into the canvas using convert_input 

    input_t = convert_input_2(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    

	# Predict the next action using the trained model	
    output = model.predict(input_t.reshape(1,-1))
    action = output[0]

    # Return the action to perform
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
