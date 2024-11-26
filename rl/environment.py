import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from PacMan.assets.constants import *
from PacMan.main import GameController


class PacManEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(PacManEnv, self).__init__()
        self.game = GameController()
        self.game.startGame()

        # Define action space (UP, DOWN, LEFT, RIGHT, STOP)
        self.action_space = spaces.Discrete(5)

        # Define observation space (Pac-Man state and map state)
        # Use a basic state representation as an example
        obs_shape = (36, 28)  # Tile dimensions of the map
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Scoring system
        self.score = 0
        self.done = False

    def reset(self):
        self.game.restartGame()
        self.done = False
        self.score = 0
        return self._get_observation()

    def _get_observation(self):
        # Example of a simple state representation
        # Could be extended for more details (e.g., ghost positions)
        state = np.zeros((36, 28), dtype=np.uint8)
        for pellet in self.game.pellets.pelletList:
            x, y = pellet.position.asInt()
            state[y // 16][x // 16] = 1  # Mark pellets on the map
        x, y = self.game.pacman.position.asInt()
        state[y // 16][x // 16] = 2  # Mark Pac-Man's position
        return state

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after done=True")

        # Map action to direction
        directions = {0: STOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT}
        self.game.pacman.direction = directions[action]

        # Update the game
        self.game.update()

        # Rewards
        reward = 0
        if self.game.pellets.numEaten:
            reward += 10  # Pellet eaten
        if len(self.game.fruitCaptured) > 0:
            reward += 50  # Fruit eaten
        if self.game.flashBG:  # Level completed
            reward += 100
        if not self.game.pacman.alive:  # Pac-Man died
            reward -= 500
            self.done = True

        # Observation
        obs = self._get_observation()

        # Check for game-over conditions
        if self.game.lives <= 0:
            self.done = True

        return obs, reward, self.done, {}

    def render(self, mode="human"):
        if mode == "human":
            self.game.render()

    def close(self):
        pass
