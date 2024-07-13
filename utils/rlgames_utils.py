
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
import torch


class ManiSkillEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        print(kwargs)
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        done = torch.logical_or(terminated, truncated)
        info["rew"] = rew
        return {"obs":obs}, rew, done, info

    def reset(self):
        out = self.env.reset()
        ot = {"obs":out[0]}
        return ot
    
    def reset_done(self):
        return self.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}

        # info["action_space"] = gym.spaces.Box(float("-inf"), float("inf"), action_shape)
        # info['observation_space'] = gym.spaces.Box(float("-inf"), float("inf"), obs_shape)
        info['observation_space'] = self.env.single_observation_space
        info['action_space'] = self.env.single_action_space

        # info['state_space'] = self.env.state_space

        # if hasattr(self.env, "amp_observation_space"):
        #     info['amp_observation_space'] = self.env.amp_observation_space

        # if self.env.num_states > 0:
        #     info['state_space'] = self.env.state_space
        # print(info['action_space'], info['observation_space'])
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


class ManiSkillAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self, n_envs):
        super().__init__()
        self.algo = None
        self.writer = None

        self.sr = 0
        self.rew = 0
        self.el_steps = 0
        self.grasp = 0
        self.g_enable = False
        self.n_envs = n_envs

        self.new_finished_episodes = False

    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), 'RLGPUAlgoObserver expects dict info'
        if "final_info" in infos:
            self.new_finished_episodes = True
            mask = infos["_final_info"]
            fin_info = infos["final_info"]
            if "success" in infos:
                # print(fin_info.keys())
                self.sr += fin_info["success"][mask].cpu().numpy().sum()
                self.rew += infos["rew"][mask].cpu().numpy().sum()
                self.el_steps += fin_info["elapsed_steps"][mask].cpu().numpy().sum()
                if "is_grasped" in fin_info:
                    self.g_enable = True
                    self.grasp += fin_info["is_grasped"][mask].cpu().numpy().sum()
            # if self.sr > 0:
            #     print("Success: ", self.sr)
            #     print("Avg Elapsed: ", fin_info["elapsed_steps"][mask].cpu().numpy().mean())
            #     print("Avg rew",infos["rew"][mask].cpu().numpy().mean())


    def after_print_stats(self, frame, epoch_num, total_time):
        if self.new_finished_episodes:
            self.writer.add_scalar("rewards/success_rate", self.sr/self.n_envs, frame)
            self.writer.add_scalar("rewards/final_reward", self.rew/self.n_envs, frame)
            self.writer.add_scalar("episode_lengths/el_steps", self.el_steps/self.n_envs, frame)
            if self.g_enable:
                self.writer.add_scalar("rewards/grasp_rate", self.grasp/self.n_envs, frame)

            self.new_finished_episodes = False
            self.sr = 0
            self.el_steps = 0
            self.rew = 0
            self.grasp = 0

