import random

import gymnasium as gym
from environment import PacmanEnvironment


class RandomAgent:
    def __init__(self, action_space):
        # L'agent choisit une action parmi les actions possibles (haut, bas, gauche, droite)
        self.action_space = action_space

    def act(self):
        # Choisir une action aléatoire parmi les actions disponibles
        return self.action_space.sample()


def train_agent():
    # Initialisation de l'environnement Pacman
    env = PacmanEnvironment()

    # Initialisation de l'agent avec l'espace d'actions de l'environnement
    agent = RandomAgent(env.action_space)

    # Nombre d'épisodes d'entraînement
    num_episodes = 10
    for episode in range(num_episodes):
        observation, info = env.reset()  # Réinitialisation de l'environnement
        done = False
        total_reward = 0

        while not done:
            # L'agent choisit une action
            action = agent.act()

            # L'agent effectue l'action dans l'environnement
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Affichage des informations de l'épisode
            print(f"Épisode {episode + 1}, Score: {total_reward}, Action: {action}")

        # Affichage du score final après chaque épisode
        print(f"Épisode {episode + 1} terminé, Score total: {total_reward}")

    # Fermeture de l'environnement après l'entraînement
    env.close()


if __name__ == "__main__":
    train_agent()
