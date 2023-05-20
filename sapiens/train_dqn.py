import os
import sys
sys.path.append(os.getcwd() )
import numpy as np
import os
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import time

BASE_PATH = "runs/debug"
GAME = "BeamRider"

LOG_PATH = os.path.join(BASE_PATH, 'random_dqn', GAME)

class MyRandomDQNAgent(dqn_agent.DQNAgent):
  def __init__(self, sess, num_actions):
    """This maintains all the DQN default argument values."""
    super().__init__(sess, num_actions)

  def step(self, reward, observation):
    """Calls the step function of the parent class, but returns a random action.
    """
    super().step(reward, observation)
    return np.random.randint(self.num_actions)

def create_random_dqn_agent(sess, environment, summary_writer=None):
  """The Runner class will expect a function of this type to create an agent."""
  return dqn_agent.DQNAgent(sess)
  #return MyRandomDQNAgent(sess, num_actions=environment.action_space.n)

def create_dqn_agent(sess, environment, summary_writer=None):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)

random_dqn_config = """
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
atari_lib.create_atari_environment.game_name = '{}'
atari_lib.create_atari_environment.sticky_actions = True
run_experiment.Runner.num_iterations = 20000
run_experiment.Runner.training_steps = 10
run_experiment.Runner.max_steps_per_episode = 100
""".format(GAME)
gin.parse_config(random_dqn_config, skip_unknown=False)

gin_files = ["dopamine/agents/dqn/configs/dqn_beamrider.gin"]
gin_bindings = []
run_experiment.load_gin_configs(gin_files, gin_bindings)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework.
#dqn_runner = run_experiment.create_runner(LOG_PATH)
dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_dqn_agent)

#runner.run()
#random_dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_random_dqn_agent)

start = time.time()
print('Will train agent, please be patient, may be a while...')
dqn_runner.run_experiment()
print('Done training!')
end = time.time()

print("Time", str(end-start))

random_dqn_data = colab_utils.read_experiment(
    LOG_PATH, verbose=True, summary_keys=['train_episode_returns'])
random_dqn_data['agent'] = 'MyDQN'
random_dqn_data['run_number'] = 1
#experimental_data[GAME] = experimental_data[GAME].merge(random_dqn_data,
 #                                                       how='outer')
experimental_data = {GAME: random_dqn_data}

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,8))
sns.lineplot(
    x='iteration', y='train_episode_returns', hue='agent',
    data=experimental_data[GAME], ax=ax)
plt.title(GAME)
plt.show()
plt.savefig(LOG_PATH + "/training.png")


# with this you can load all log data. this is the data you saved using Statistics
raw_data, _ = colab_utils.load_statistics(LOG_PATH + "/logs", verbose=False)

summarized_data = colab_utils.summarize_data(
      raw_data, ['train_episode_returns'])
plt.plot(summarized_data['train_episode_returns'], label='episode returns')
plt.plot()
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.legend()
plt.show()

# gin.clear_config()
# run_experiment.load_gin_configs(gin_files, gin_bindings)
#
#
# from dopamine.utils import example_viz_lib
# num_steps = 1000  # @param {type:"number"}
# example_viz_lib.run(agent='MyDQN', game=GAME, num_steps=num_steps,
#                     root_dir=LOG_PATH, restore_ckpt=LOG_PATH + '/checkpoints/tf_ckpt-9',
#                     use_legacy_checkpoint=True)
# print("check")