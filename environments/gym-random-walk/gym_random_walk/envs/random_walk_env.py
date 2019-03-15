import sys
from string import ascii_uppercase

import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from six import StringIO

LEFT, RIGHT = 0, 1


class RandomWalkEnv(discrete.DiscreteEnv):
	metadata = {'render.modes': ['human', 'ansi']}
	
	def __init__(self, n_states=7):
		
		self.shape = (1, n_states)
		self.start_state_index = self.shape[1] // 2
		
		self.nS = nS = n_states
		self.nA = nA = 2
		
		p = {}
		for s in range(self.nS):
			p[s] = {}
			for a in range(nA):
				prob = 1.0
				new_state = np.clip(s - 1 if a == LEFT else s + 1, 0, nS - 1)
				new_state = s if s == 0 or s == nS - 1 else new_state
				reward = 1.0 if new_state == nS - 1 and s != new_state else 0.0
				done = new_state == 0 or new_state == nS - 1
				p[s][a] = [(prob, new_state, reward, done)]
		
		isd = np.zeros(nS)
		isd[self.start_state_index] = 1.0
		
		discrete.DiscreteEnv.__init__(self, nS, nA, p, isd)
	
	def render(self, mode='human', close=False):
		outfile = StringIO() if mode == 'ansi' else sys.stdout
		desc = np.asarray([ascii_uppercase[:self.shape[1]]], dtype='c').tolist()
		desc = [[c.decode('utf-8') for c in line] for line in desc]
		color = 'red' if self.s == 0 else 'green' if self.s == self.nS - 1 else 'yellow'
		desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
		outfile.write("\n")
		outfile.write("\n".join(''.join(line) for line in desc) + "\n")
		
		if mode != 'human':
			return outfile
