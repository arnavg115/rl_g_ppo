import gym.spaces
from rl_games.common import env_configurations, vecenv
import torch
import gym


class ManiSkillEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print(kwargs)
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        done = torch.logical_or(terminated, truncated)
        return {"obs":obs}, rew, done, info

    def reset(self):
        return {"obs":self.env.reset()[0]}
    
    def reset_done(self):
        return self.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        action_shape = self.env.single_action_space.shape
        obs_shape = self.env.single_observation_space.shape

        info["action_space"] = gym.spaces.Box(float("-inf"), float("inf"), action_shape)
        info['observation_space'] = gym.spaces.Box(float("-inf"), float("inf"), obs_shape)
        # info['observation_space'] = self.env.single_observation_space
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