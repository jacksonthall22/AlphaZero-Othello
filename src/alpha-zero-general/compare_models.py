import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import torch
import matplotlib.pyplot as plt


import numpy as np
from utils import dotdict, NetworkType, NETWORK_TYPE

"""
use this script to play the cnn networks against the vit networks
"""
NUM_GAMES = 1000
all_stats = torch.zeros((10, 3))
mini_othello = True  # Play in 6x6 instead of the normal 8x8.
print("(CNN Wins, ViT Wins, Draws)")
for iteration in range(10, 101, 10):
    if mini_othello:
        g = OthelloGame(6)
    else:
        g = OthelloGame(8)

    n1 = NNet(g, NetworkType.CNN)
    found = False
    curr = iteration
    while not found:
        try:
            n1.load_checkpoint('../../trained_models/', "cnn_checkpoint_" + str(curr) + ".pth.tar")
            found = True
        except RuntimeError as e: #If the checkpoint does not exist, then a previous model was considered to be better, so use that one
            curr -= 1
    args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


    n2 = NNet(g, NetworkType.VIT)
    found = False
    curr = iteration
    while not found:
        try:
            n2.load_checkpoint('../../trained_models/', "vit_checkpoint_" + str(curr) + ".pth.tar")
            found = True
        except RuntimeError: #If the checkpoint does not exist, then a previous model was considered to be better, so use that one
            curr -= 1
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

    print("Iteration: " + str(iteration))
    stats = arena.playGames(NUM_GAMES, verbose=False)
    print(stats)
    all_stats[(iteration // 10) - 1] = torch.Tensor(stats)

print(all_stats)
cnn_wins = all_stats[:, 0]
vit_wins = all_stats[:, 1]
labels = list(range(10, 101, 10))
plt.bar(labels, cnn_wins, width=8, color="blue", label="CNN")
plt.bar(labels, vit_wins, width=8, color="orange", label="ViT", bottom=cnn_wins)
plt.ylabel("Wins")
plt.xlabel("Iterations Trained")
plt.title("CNN vs ViT in Othello 6x6")
plt.legend()
plt.savefig("../../compared_models_" + str(NUM_GAMES) + "_games.png", dpi=300)
