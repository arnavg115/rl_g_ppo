

import hydra

from omegaconf import DictConfig, OmegaConf


# def preprocess_train_config(cfg, config_dict):
#     """
#     Adding common configuration parameters to the rl_games train config.
#     An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
#     variable interpolations in each config.
#     """

#     train_cfg = config_dict['params']['config']

#     train_cfg['device'] = cfg.rl_device

#     train_cfg['population_based_training'] = cfg.pbt.enabled
#     train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

#     train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

#     print(f'Using rl_device: {cfg.rl_device}')
#     print(f'Using sim_device: {cfg.sim_device}')
#     print(train_cfg)

#     try:
#         model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
#         if model_size_multiplier != 1:
#             units = config_dict['params']['network']['mlp']['units']
#             for i, u in enumerate(units):
#                 units[i] = u * model_size_multiplier
#             print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
#     except KeyError:
#         pass

#     return config_dict



@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    import os
    import mani_skill.envs
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from datetime import datetime
    import gymnasium as gym
    # from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from utils.rlgames_utils import ManiSkillEnv
    # from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from utils.utils import set_seed
    from utils.reformat import omegaconf_to_dict, print_dict
    # from isaacgymenvs.learning import amp_continuous
    # from isaacgymenvs.learning import amp_players
    # from isaacgymenvs.learning import amp_models
    # from isaacgymenvs.learning import amp_network_builder
    # import isaacgymenvs


    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    # if cfg.checkpoint:
    #     cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)


    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_maniskill_env(**kwargs):
        print(cfg.task.env)
        envs = gym.make(
            "PickCube-v1",
           **cfg.task.env
        )
        envs = ManiSkillVectorEnv(envs)
        return envs

    env_configurations.register('maniskill', {
        'vecenv_type': 'MANISKILL',
        'env_creator': lambda **kwargs: create_maniskill_env(**kwargs),
    })



    vecenv.register('MANISKILL', lambda config_name, num_actors, **kwargs: ManiSkillEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    # rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)


    # register new AMP network builder and agent
    # def build_runner(algo_observer):
    #     runner = Runner(algo_observer)
    #     runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
    #     runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    #     model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
    #     model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

    #     return runner

    # # convert CLI arguments into dictionary
    # # create runner and set the settings
    def build_runner():
        runner = Runner()
        return runner

    runner = build_runner()
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    # if not cfg.test:
    #     experiment_dir = os.path.join('runs', cfg.train.params.config.name + 
    #     '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

    #     os.makedirs(experiment_dir, exist_ok=True)
    #     with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
    #         f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()