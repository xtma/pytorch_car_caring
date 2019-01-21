# Car Racing with PyTorch
Solving the car racing problem in OpenAI Gym using Proximal Policy Optimization (PPO). This problem has a real physical engine in the back end. You can achieve real racing actions in the environment, like drifting. 

## Requirement
To run the code, you need
- [pytorch 0.4](https://pytorch.org/)
- [gym 0.10](https://github.com/openai/gym)
- [visdom 0.1.8](https://github.com/facebookresearch/visdom)

## Method
Every action will be repeated for 8 frames. To get velocity information, state is defined as adjacent 4 frames in shape (4, 96, 96). Use a two heads FCN to represent the actor and critic respectively. The actor outputs α, β for each actin as the parameters of Beta distribution. 
<div align=center><img src="img/network.png" width="30%" /></div>

## Training
Start a Visdom server with ```python -m visdom.server```, it will serve http://localhost:8097/ by default.

To train the agent, run```python train.py --render --vis``` or ```python train.py --render``` without visdom. 
To test, run ```python test.py --render```.

## Performance
<div align=center><img src="img/car_racing_ppo.png" width="50%"/></div>
<div align=center><img src="img/car_racing_demo_ppo.gif"/></div>

