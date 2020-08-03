# Reinforcement-Learning

### Setup
Requirements: python - 3.7, tensorflow - 2.2.0 <br>
Clone the repository and install dependencies from requirements.txt <br>

        $ git clone https://github.com/DheerajRacha/Reinforcement-Learning.git
        $ cd Reinforcement-Learning
        $ pip install -r requirements.txt

### Train RL Agent
Choose a Reinforcement learning agent from *Agents* folder. <br>
Choose an OpenAIGym environment and give its ID and number of states as input 
to instance of type of agent chosen.<br>

        $ python
        >>> from Agents.AgentDQN import AgentDQN
        >>> agent = AgentDQN(env="MountainCar-v0", num_state=2)
        >>> agent.train_agent(num_episodes=1000, target_update=15) 

Trained DQN and DDQN on "MountainCar-v0" environment and corresponding checkpoints 
are provided in *checkpoints* folder

### Inference

        $ python
        >>> from Agents.AgentDQN import AgentDQN
        >>> agent = AgentDQN(env="MountainCar-v0", num_state=2)
        >>> agent.test_agent(checkpoints_path="checkpoints/DQN_MountainCar/ckpt_1000.h5", num_episodes=1)

The above code snippet renders this window. <br>

![DQN Agent](GIFs/MountainCar_DQN_animation.gif)

### ToDos

- [ ] Q-Learning
- [ ] SARSA
- [x] DQN
- [x] DDQN
- [ ] Policy Gradients
- [ ] Actor-Critics

