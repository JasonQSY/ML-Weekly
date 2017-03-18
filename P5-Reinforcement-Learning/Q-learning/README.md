# Q-Learning-Algorithm

Q-learning is a reinforcement learning algorithm.

## Basic

- [Q-learning, Wikipedia](https://en.wikipedia.org/wiki/Q-learning)
- [A painless q learning tutorial](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)

## Results

- [Q-Learning-Get-Started.ipynb](Q-Learning-Get-Started.ipynb)

## Problem

> Q-learning at its simplest uses tables to store data. This very quickly loses viability with increasing sizes of state/action space of the system it is monitoring/controlling. One answer to this problem is to use an (adapted) artificial neural network as a function approximator, as demonstrated by Tesauro in his Backgammon playing temporal difference learning research.

> More generally, Q-learning can be combined with function approximation.[8] This makes it possible to apply the algorithm to larger problems, even when the state space is continuous, and therefore infinitely large. Additionally, it may speed up learning in finite problems, due to the fact that the algorithm can generalize earlier experiences to previously unseen states.

## Solution

Use a function approximation.

- Linear
- neuron network

and can be implemented by `tensorflow`.

## Related Resources

- [Temporal Difference Learning and TD-Gammon](http://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf)
- [Questions about Q-Learning using Neural Networks](http://stats.stackexchange.com/questions/126994/questions-about-q-learning-using-neural-networks)
- [Q-learning with Neural Network as function approximation](http://stats.stackexchange.com/questions/200006/q-learning-with-neural-network-as-function-approximation)
