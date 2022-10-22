class MyModel:
    def predict(self, env, observation, agent):
        action = env.action_space(agent).sample()
        return action
