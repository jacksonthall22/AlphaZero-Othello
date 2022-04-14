# coding=utf-8
import logging

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import dotdict, NetworkType, NETWORK_TYPE

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def main():
    if NETWORK_TYPE == NetworkType.CNN:
        num_iters = 2
        load_model = True
        load_folder_file = 'cnn_best.pth.tar'
    elif NETWORK_TYPE == NetworkType.VIT:
        num_iters = 1
        load_model = True
        load_folder_file = 'vit_best.pth.tar'
    else:
        raise ValueError(f'NETWORK_TYPE="{network_type}" not yet implemented')

    args = dotdict({
        'network_type': NETWORK_TYPE,

        'numIters': num_iters,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': load_model,
        'load_folder_file': ('temp', load_folder_file),
        'numItersForTrainExamplesHistory': 20,

    })

    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, NETWORK_TYPE)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    # print('====================================')
    # print('            TRAINING CNN            ')
    # print('====================================')
    # NETWORK_TYPE = NetworkType.CNN
    # main()

    print('====================================')
    print('            TRAINING VIT            ')
    print('====================================')
    NETWORK_TYPE = NetworkType.VIT
    main()
