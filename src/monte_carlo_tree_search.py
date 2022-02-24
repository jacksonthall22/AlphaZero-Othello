"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

LINK TO THIS CODE:
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from typing import Reversible, Generator, List, Set
import chess.pgn
import numpy as np


class MCTS:
    """ Monte Carlo tree searcher. First rollout the tree then choose a move. """

    def __init__(self, exploration_weight: int = 1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()     # children of each node (map the hash to the Node)
        self.exploration_weight = exploration_weight

    def choose(self, node: 'Node'):
        """ Choose the best successor of node (choose a move in the game). """
        if node.is_terminal():
            raise RuntimeError(f"MCTS.choose() called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node: 'Node') -> None:
        """ Make the tree one layer better (train for one iteration). """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node: 'Node') -> List['Node']:
        """ Find an unexplored descendant of `node` """
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node: 'Node') -> None:
        """ Update the `children` dict with the children of `node`. """
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node: 'Node') -> float:
        """ Returns the reward for a random simulation (to completion) of `node`. """
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path: Reversible['Node'], reward: float) -> None:
        """ Send the reward back up to the ancestors of the leaf """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node: 'Node') -> 'Node':
        """ Select a child of node, balancing exploration & exploitation """

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n: Node):
            """ Upper confidence bound for trees """
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self) -> Set['Node']:
        """ All possible successors of this board state """
        return set()

    @abstractmethod
    def find_random_child(self) -> 'Node':
        """ Random successor of this board state (for more efficient simulation) """
        return self

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Returns True if the node has no children """
        return True

    @abstractmethod
    def reward(self) -> float:
        """ Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc """
        return 0

    @abstractmethod
    def __hash__(self) -> int:
        """ Nodes must be hashable """
        return 123456789

    @abstractmethod
    def __eq__(self, other: 'Node') -> bool:
        """ Nodes must be comparable """
        return True


class MCTSGameNode(Node):

    def __init__(self, game_node: chess.pgn.ChildNode):
        self.game_node = game_node

    def find_children(self) -> Set['Node']:
        return set(MCTSGameNode(self.game_node.add_variation(move))
                   for move in self.game_node.board().generate_legal_moves())

    def find_random_child(self) -> 'Node':
        return np.random.choice(self.find_children())

    def is_terminal(self) -> bool:
        return self.game_node.board().is_game_over()

    def reward(self) -> float:
        # todo Reward estimation will be the policy's output for this position.
        #      Policy will be trained to make its output for each position (static eval)
        #      match the reward found through MCTS minimax (dynamic eval, a much better estimate)
        return 0

    def __hash__(self) -> int:
        return hash(self.game_node)

    def __eq__(self, other: 'Node') -> bool:
        if not isinstance(other, MCTSGameNode):
            return False

        return self == other.game_node
