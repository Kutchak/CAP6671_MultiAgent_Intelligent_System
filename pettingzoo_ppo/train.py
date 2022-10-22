import gym
from pettingzoo.magent import combined_arms_v5
import gnwrapper
import pickle
from model import MyModel
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    MAX_TIMESTEPS = 500

    # Initialize the Combined Arms environment
    env = combined_arms_v5.parallel_env() # Parallel environemnt
    env = gnwrapper.LoopAnimation(env) # Start Xvfb
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

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
