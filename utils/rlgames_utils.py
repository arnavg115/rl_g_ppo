from rl_games.common import env_configurations, vecenv


class ManiSkillEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print(kwargs)
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        # info['state_space'] = self.env.state_space

        # if hasattr(self.env, "amp_observation_space"):
        #     info['amp_observation_space'] = self.env.amp_observation_space

        # if self.env.num_states > 0:
        #     info['state_space'] = self.env.state_space
        print(info['action_space'], info['observation_space'])
        # else:
        #     print(info['action_space'], info['observation_space'])

        return info

    def set_train_info(self, env_frames, *args_, **kwargs_):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        if hasattr(self.env, 'set_train_info'):
            self.env.set_train_info(env_frames, *args_, **kwargs_)