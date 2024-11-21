import gym

from rl.environment import PacManEnv

# Initialisation de l'environnement
env = PacManEnv()

# Boucle d'interaction
state = env.reset()
done = False

try:
    while not done:
        action = env.action_space.sample()  # Action aléatoire
        next_state, reward, done, info = env.step(action)
        env.render()  # Affiche l'état actuel du jeu
finally:
    env.close()  # Nettoie les ressources
