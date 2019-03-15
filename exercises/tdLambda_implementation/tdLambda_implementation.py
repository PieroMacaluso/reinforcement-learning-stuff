import random
import sys
import itertools

import gym
import gym_random_walk
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

env = gym.make('RandomWalkSeven-v0')


def td_lambda_prediction(policy, env, num_episodes, discount_factor=1.0, alpha=1.0, lamb=0.0):
	"""
	TD(lambda) prediction algorithm. Calculates the value function
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
	:param lamb:
			Lambda parameter
	:return:
	"""
	n_state = env.env.nS
	v = np.ones(n_state) / 2
	v[0] = 0.0
	v[n_state - 1] = 0.0
	eligibility = np.zeros(n_state)
	
	# Run the algorithm for some episodes
	for i_episode in range(1, num_episodes + 1):
		# Print out which episode we're on, useful for debugging.
		if i_episode % 1000 == 0:
			print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
			sys.stdout.flush()
		
		# Generate an episode.
		# An episode is an array of (state, action, reward) tuples
		state = env.reset()
		for t in itertools.count():
			# act according to policy
			action = policy(state)
			new_state, reward, done, _ = env.step(action)
			# Update eligibilities
			eligibility *= lamb * discount_factor
			eligibility[state] += 1.0
			
			# get the td-error and update every state's value estimate
			# according to their eligibilities.
			td_error = reward + discount_factor * v[new_state] - v[state]
			v = v + alpha * td_error * eligibility
			
			if done:
				break
			state = new_state
	
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
		x = range(0, n_state - 2)
		y = [x / (n_state - 1) for x in range(1, n_state - 1)]
		ax.plot(x, y, marker='o', label='True values')
		
		leg = ('1 episode', '10 episodes', '100 episodes')
		labels = [chr(idx + 65) for idx, val in enumerate(range(1, n_state - 1))]
		
		for i_v, l in zip(v, leg):
			ax.plot(labels, i_v[1:(n_state - 1)], marker='o', label=l)
		
		ax.set_xlabel('State')
		ax.set_ylabel('Estimated Value')
		ax.set_title(graph_title)
		# Only integer ticks
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# One tick every integer
		plt.xticks(np.arange(0, n_state - 1, 1.0))
		# Plot legend
		plt.legend()
		plt.grid()
		plt.show()
	
	plot_scatter(v, title)


# HyperParameters
alpha = 0.1
lamb = 0.3

v_1 = td_lambda_prediction(uniform_left_right_policy, env, num_episodes=1, alpha=alpha, lamb=lamb)
v_10 = td_lambda_prediction(uniform_left_right_policy, env, num_episodes=10, alpha=alpha, lamb=lamb)
v_100 = td_lambda_prediction(uniform_left_right_policy, env, num_episodes=100, alpha=alpha, lamb=lamb)


plot_random_walk((v_1, v_10, v_100), f"Random Walk TD (Lambda) with {env.env.nS} states")

pass
