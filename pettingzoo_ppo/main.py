import gym
from pettingzoo.magent import combined_arms_v5
import gnwrapper
import pickle
from model import MyModel

MAX_TIMESTEPS = 500

# Initialize the Combined Arms environment
env = combined_arms_v5.parallel_env() # Parallel environemnt
env = gnwrapper.LoopAnimation(env) # Start Xvfb

# Instantiate models
redModel = MyModel()
blueModel = MyModel()

# Save (pickle) model
pickle.dump(redModel, open("redModel.pkl", 'wb') )

# Load pickled model
with open("redModel.pkl", 'rb') as pickle_model:
  redModel = pickle.load(pickle_model)

observations = env.reset()

for step in range(MAX_TIMESTEPS):
  actions = {}

  # For each agent, get the next action
  for agent in env.agents:
    model = None
    if "red" in agent:
      model = redModel
    elif "blue" in agent:
      model = blueModel

    actions.update({agent: model.predict(env, observations[agent], agent) })

  observations, rewards, dones, infos = env.step(actions)

  env.render() # Render the current frame

# Display the rendered movie
env.display()

# Final Stats
redMelee = 0
redRanged = 0
blueMelee = 0
blueRanged = 0
for agent in env.agents:
  if "redmelee" in agent:
    redMelee += 1
  elif "redranged" in agent:
    redRanged += 1
  elif "bluemele" in agent:
    blueMelee += 1
  elif "blueranged" in agent:
    blueRanged += 1

print("Red Total {}: (Melee {}, Ranged {})".format(redMelee+redRanged,redMelee, redRanged) )
print("Blue Total {}: (Melee {}, Ranged {})".format(blueMelee+blueRanged,blueMelee, blueRanged) )
