import time

import gym
import numpy as np
from gym import spaces

from PacMan_Game.main import GameController


class PacManEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(PacManEnv, self).__init__()
        self.game = GameController()
        self.game.startGame()
        self.action_space = spaces.Discrete(4)

        # Définir l'espace d'observation (par exemple, une image ou des informations symboliques)
        # Exemple avec un vecteur d'observation symbolique
        # (Pac-Man position, Fantômes positions, état des pac-dots)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.get_state_shape(),),
            dtype=np.float32,
        )

        self.total_reward = 0  # Variable to accumulate the total reward during the run

    def reset(self):
        """
        Réinitialise l'environnement et renvoie l'état initial.
        """
        self.game.restartGame()
        self.total_reward = 0  # Reset the total reward for the new run
        return self.get_state()

    def step(self, action):
        """
        Applique une action et retourne (observation, récompense, done, info).
        """
        # Translate the action into Pac-Man movement
        self.game.pacman.change_direction(
            action
        )  # Ensure change_direction method exists

        # Add a small delay between actions to avoid excessive speed
        time.sleep(0.1)  # Adjust the delay as necessary

        # Update the game
        self.game.update()

        # Get the current state
        observation = self.get_state()

        # Calculate the reward
        reward = self.calculate_reward()

        # Update the total reward
        self.total_reward += reward

        # Check if the game is done
        done = not self.game.pacman.alive or self.game.lives == 0

        # Additional information
        info = {"score": self.game.score}

        if done:
            # Log the total reward at the end of the run
            print(f"End of game. Total reward: {self.total_reward}")

        return observation, reward, done, info

    def render(self, mode="human"):
        """
        Affiche l'état actuel du jeu.
        """
        if mode == "human":
            self.game.render()

    def close(self):
        """
        Ferme l'environnement.
        """
        self.game = None

    def get_state(self):
        """
        Retourne l'état actuel sous forme d'un vecteur ou d'une matrice.
        Exemple : positions, pac-dots restants, etc.
        """
        # Exemple simplifié : position de Pac-Man, positions des fantômes
        pacman_pos = self.game.pacman.position
        ghost_positions = [ghost.position for ghost in self.game.ghosts]
        num_pellets = len(self.game.pellets.pelletList)
        return np.array(
            [pacman_pos.x, pacman_pos.y]
            + [g.x for g in ghost_positions]
            + [g.y for g in ghost_positions]
            + [num_pellets]
        )

    def get_state_shape(self):
        if not hasattr(self.game, "ghosts"):
            raise AttributeError(
                "Les fantômes ne sont pas initialisés dans GameController."
            )
        num_ghosts = len(self.game.ghosts.ghosts)  # Corrected this line
        return 2 + 2 * num_ghosts + 1

    def calculate_reward(self):
        """
        Calcule une récompense en fonction des événements dans le jeu.
        Exemple :
        - +10 pour un pellet mangé
        - +50 pour un fruit
        - -100 pour être capturé par un fantôme
        """
        reward = 0
        if not self.game.pacman.alive:
            reward -= 100  # Perte de vie
        if self.game.pellets.numEaten > 0:
            reward += 10 * self.game.pellets.numEaten
            self.game.pellets.numEaten = 0  # Réinitialiser le compteur
        return reward
