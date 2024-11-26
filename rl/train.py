import gymnasium as gym
from agent import RandomAgent
from environment import PacManEnv

if __name__ == "__main__":
    import gymnasium as gym
    from environment import PacManEnv

    env = PacManEnv()
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")
    env.close()
