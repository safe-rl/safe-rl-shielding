Below, one can find the instructions to install all dependencies. 

## Usage for Seaquest:
 - **Without shield**: python main.py --env_name=Seaquest-v0 --is\_train=True --use\_gpu=True --display=False --model=m2
 - **With shield and original rewards for all actions**: python main.py --env_name=Seaquest-v0 --is\_train=True --use\_gpu=True --display=False --model=m2 --shield\_active=True
 - **With shield and punishment for unsafe actions**: python main.py --env_name=Seaquest-v0 --is\_train=True --use\_gpu=True --display=False --model=m2 --shield\_active=True --negative\_reward=True

The **gen\_seaquest\_spec** script can be used to create the specification file **seaquest.dfa**. The shield synthesizer tool can be used to create the **seaquest.py** file, which contains the shield code, but that can take a few minutes.

# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

![model](assets/model.png)

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learnig targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values


## Requirements

- Python 2.7 or Python 3.3+
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)


## Usage

First, install prerequisites with:

    $ pip install tqdm gym[all]

To train a model for Breakout:

    $ python main.py --env_name=Breakout-v0 --is_train=True
    $ python main.py --env_name=Breakout-v0 --is_train=True --display=True

To test and record the screen with gym:

    $ python main.py --is_train=False
    $ python main.py --is_train=False --display=True


## Results

Result of training for 24 hours using GTX 980 ti.

![best](assets/best.gif)


## Training details

Details of `Breakout` with model `m2`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m2.png)

Details of `Breakout` with model `m3`(red) for 30 hours using GTX 980 Ti.

![tensorboard](assets/0620_scalar_step_m3.png)


## References

- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
