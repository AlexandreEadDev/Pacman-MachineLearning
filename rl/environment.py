import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gym
import numpy as np
from gym import spaces

from PacMan.assets.constants import *
from PacMan.fruit import Fruit
from PacMan.ghosts import GhostGroup
from PacMan.mazedata import MazeData
from PacMan.pacman import Pacman
from PacMan.pellets import PelletGroup
from PacMan.vector import Vector2


class PacmanEnvironment(gym.Env):
    def __init__(self):
        super(PacmanEnvironment, self).__init__()

        # Définir l'espace d'action (haut, bas, gauche, droite)
        self.action_space = spaces.Discrete(4)  # 4 actions : UP, DOWN, LEFT, RIGHT

        # Définir l'espace d'état (composé de positions de Pac-Man, des fantômes, score)
        # L'état contient :
        # - Position de Pac-Man (x, y)
        # - Positions des fantômes
        # - Nombre de pellets mangés
        # - Score actuel
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(10,), dtype=np.float32
        )

        self.level = 0
        self.score = 0
        self.lives = 5
        self.pacman = None
        self.ghosts = None
        self.pellets = None
        self.fruit = None
        self.mazedata = MazeData()

    def reset(self):
        """Réinitialiser l'environnement pour commencer une nouvelle partie"""
        self.level = 0
        self.score = 0
        self.lives = 5
        self.mazedata.loadMaze(self.level)
        self.pacman = Pacman(self.mazedata.obj.pacmanStart)
        self.ghosts = GhostGroup(self.pacman.node, self.pacman)
        self.pellets = PelletGroup(self.mazedata.obj.name + ".txt")
        self.fruit = None

        return self.get_state()

    def get_state(self):
        """Retourne l'état sous forme de tableau avec des informations utiles"""
        pacman_position = np.array([self.pacman.position.x, self.pacman.position.y])
        ghost_positions = np.array(
            [ghost.position.x, ghost.position.y] for ghost in self.ghosts
        )
        pellet_count = len(self.pellets.pelletList)
        score = np.array([self.score])

        # Combine les informations dans un seul tableau d'état
        state = np.concatenate(
            [pacman_position.flatten(), ghost_positions.flatten(), score]
        )
        return state

    def step(self, action):
        """Appliquer l'action et renvoyer l'état suivant, la récompense, et les informations"""
        self.handle_action(action)

        # Mettre à jour l'environnement
        self.update()

        # Vérifier si Pac-Man est mort ou si le niveau est terminé
        done = self.is_done()

        # Récompenser
        reward = self.calculate_reward()

        return self.get_state(), reward, done, {}

    def handle_action(self, action):
        """Gérer l'action (mouvement de Pac-Man)"""
        if action == 0:  # UP
            self.pacman.move(UP)
        elif action == 1:  # DOWN
            self.pacman.move(DOWN)
        elif action == 2:  # LEFT
            self.pacman.move(LEFT)
        elif action == 3:  # RIGHT
            self.pacman.move(RIGHT)

    def update(self):
        """Mettre à jour l'environnement, déplacer les fantômes, etc."""
        self.pacman.update(1.0 / 60.0)  # Exemple avec un dt de 1/60s
        self.ghosts.update(1.0 / 60.0)
        if self.fruit is not None:
            self.fruit.update(1.0 / 60.0)

        # Vérifier les collisions et mettre à jour le score
        self.check_collisions()

    def check_collisions(self):
        """Vérifier si Pac-Man mange un pellet, un fruit, ou entre en collision avec un fantôme"""
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.pelletList.remove(pellet)
            self.score += pellet.points

        if self.fruit is not None and self.pacman.collideCheck(self.fruit):
            self.score += self.fruit.points
            self.fruit = None

        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current != FREIGHT:
                    self.lives -= 1
                    self.pacman.die()
                    self.ghosts.reset()
                    if self.lives <= 0:
                        self.score -= 500  # Récompense négative pour la mort
                        return

    def is_done(self):
        """Vérifie si l'épisode est terminé (Pac-Man mort ou niveau terminé)"""
        if self.lives <= 0:
            return True
        if self.pellets.isEmpty():
            self.level += 1
            self.mazedata.loadMaze(self.level)
            self.reset()
        return False

    def calculate_reward(self):
        """Calculer la récompense pour l'agent"""
        reward = 0
        # Ajouter la récompense pour chaque pellet ou fruit mangé
        if self.pellets.isEmpty():
            reward += 100  # Récompense pour avoir terminé un niveau
        if self.pacman.alive:
            return reward
        else:
            return -500  # Récompense négative si Pac-Man meurt

    def render(self):
        """Rendre l'état de l'environnement (peut être utilisé pour la visualisation)"""
        pass  # Pas nécessaire si non utilisé
