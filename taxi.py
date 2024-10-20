"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.3, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4), render=False):
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    frames = []
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r

        if render:
            frames.append(env.render())

        s = next_s
        if done:
            break
        # END SOLUTION

    return total_reward, frames


def create_animation(frames: t.List[np.ndarray], title: str):
    fig, ax = plt.subplots()
    plt.axis('off')

    frames = [[plt.imshow(frame)] for frame in frames]

    animation_fig = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=1000)
    plt.show()

    animation_fig.save(f"videos/{title}_{time.time()}.gif")


rewards = []
for i in range(1000):
    reward, frames = play_and_train(env, agent, render=i % 100 == 0)
    rewards.append(reward)
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        create_animation(frames, f"qlearning-train{i}")

assert np.mean(rewards[-100:]) > 0.0

x = np.linspace(0, 1000, 1000)
plt.scatter(x, rewards)
plt.title("Q-Learning rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.savefig("qlearning-rewards.png")

create_animation(play_and_train(env, agent, t_max=200, render=True)[1], "qlearning-final")

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    reward, frames = play_and_train(env, agent, render=i % 100 == 0)
    rewards.append(reward)
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        create_animation(frames, f"qlearning-eps-train{i}")

assert np.mean(rewards[-100:]) > 0.0

plt.scatter(x, rewards)
plt.title("Q-Learning Eps Scheduling rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.savefig("qlearning-eps-rewards.png")

create_animation(play_and_train(env, agent, t_max=200, render=True)[1], "qlearningsEpsScheduling")

####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    reward, frames = play_and_train(env, agent, render=i % 100 == 0)
    rewards.append(reward)
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        create_animation(frames, f"sarsa-train{i}")


plt.scatter(x, rewards)
plt.title("Q-Learning rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.savefig("qlearning-rewards.png")

create_animation(play_and_train(env, agent, t_max=200, render=True)[1], "sarsa")
