# Gym Random Walk Environment

OpenAI Gym Environment of Random Walk example provided in [1] Chapter 6, example 6.2

![RandomWalk](https://i.imgur.com/nHgMGr1.png)
## Installation

>Assuming you are in rl-exercise folder and using **pipenv**

```bash
pipenv install environment/gym-random-walk
```
## Usage Example
```python
import gym
import gym_random_walk

env = gym.make('RandomWalkSeven-v0')
```


## References
[1] Richard S. Sutton, Andrew G. Barto - [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf)