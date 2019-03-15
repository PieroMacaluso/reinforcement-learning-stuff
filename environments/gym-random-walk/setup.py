from setuptools import setup

setup(
    name='gym_random_walk',
    version='0.0.1',
    description='Gym Random Walk Environment',
    url='https://github.com/pieromacaluso/gym-random-walk',
    author='Piero Macaluso',
    author_email='pieromacaluso8@gmail.com',
    packages=['gym_random_walk', 'gym_random_walk.envs'],
    install_requires=['gym'],
)
