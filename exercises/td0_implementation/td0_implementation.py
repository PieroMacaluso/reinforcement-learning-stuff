import random
import sys
import itertools

import gym
import gym_walk
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

env = gym.make('WalkSevenStates-v0')


def td0_prediction(policy, env, num_episodes, discount_factor=1.0, alpha=1.0):
	"""
	TD(0) prediction algorithm. Calculates the value function
	for a given policy.

	:param policy:
		A function that maps an observation to action probabilities.
	:param env:
		OpenAI gym environment.
	:param num_episodes:
		Number of episodes.
	:param discount_factor:
		Gamma discount factor.
	:param alpha:
		Step size.
	:return:
		A dictionary that maps from state -> value.
		The state is an integer and the value is a float.
	"""
	n_state = env.env.nS
	# The final value function
	# The starting value is 0.5 for each non-terminal element, otherwise 0
	v = np.ones(n_state)/2
	v[0] = 0.0
	v[n_state - 1] = 0.0
	
	for i_episode in range(1, num_episodes + 1):
		# Print out which episode we're on, useful for debugging.
		if i_episode % 1000 == 0:
			print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
			sys.stdout.flush()
		
		# Generate an episode.
		# An episode is an array of (state, action, reward) tuples
		state = env.reset()
		for t in itertools.count():
			action = policy(state)
			next_state, reward, done, _ = env.step(action)
			v[state] += alpha * (reward + discount_factor * v[next_state] - v[state])
			if done:
				break
			state = next_state
	
	return v


def uniform_left_right_policy(observation):
	"""
	A policy where move left and move right have the same probability
	"""
	return random.randint(0, 1)


def plot_random_walk(v, title="Value Function"):
	"""
	Plots the value function as a surface plot.
	"""
	
	def plot_scatter(v, graph_title):
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(111)
		n_state = len(v[0])
		x = range(0, n_state-2)
		y = [x / (n_state-1) for x in range(1, n_state-1)]
		ax.plot(x, y, marker='o', label='True values')
		
		leg = ('1 episode', '10 episodes', '100 episodes')
		labels = [chr(idx + 65) for idx, val in enumerate(range(1, n_state-1))]

		for i_v, l in zip(v, leg):
			ax.plot(labels, i_v[1:(n_state-1)], marker='o', label=l)
		
		ax.set_xlabel('State')
		ax.set_ylabel('Estimated Value')
		ax.set_title(graph_title)
		# Only integer ticks
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# One tick every integer
		plt.xticks(np.arange(0, n_state-1, 1.0))
		# Plot legend
		plt.legend()
		plt.grid()
		plt.show()
	
	plot_scatter(v, title)


v_1 = td0_prediction(uniform_left_right_policy, env, num_episodes=1, alpha=0.1)
v_10 = td0_prediction(uniform_left_right_policy, env, num_episodes=10, alpha=0.1)
v_100 = td0_prediction(uniform_left_right_policy, env, num_episodes=100, alpha=0.1)

plot_random_walk((v_1, v_10, v_100), f"Random Walk TD (0) with {env.env.nS} states")

pass
