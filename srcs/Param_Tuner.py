
import numpy as np 

from omegaconf import DictConfig, OmegaConf
import srcs.Trainer as trainer

## Hyperparameter Tuning
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import air
from ray.air import session

def _hyperparameter_tune(config, cfg):
    
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict["model"].update((k, config[k]) for k in cfg_dict["model"].keys() & config.keys())
    cfg_dict["dataset"].update((k, config[k]) for k in cfg_dict["dataset"].keys() & config.keys())
    
    cfg = OmegaConf.create(cfg_dict)
    
    return trainer.train_FX(cfg, tuning_mode = True)


def run(cfg: DictConfig):
    
    ## Hyperparameter Search
    num_samples = cfg.dataset.hyperparameter_ray_num_samples

    config    = {   "d_ff"      : tune.sample_from(lambda _: 2 ** np.random.randint(5, 10)),
                    "n_heads"   : tune.sample_from(lambda _: 2 ** np.random.randint(2, 6)),
                    "d_model"   : tune.sample_from(lambda _: 2 ** np.random.randint(5, 10)),
                    "e_layers"  : tune.sample_from(lambda _: 2 ** np.random.randint(0, 3)),
                    "lr"        : tune.loguniform(1e-4, 1e-6),
                    "patch_len" : tune.choice([3, 6, 10, 15, 20, 30, 60])}

    scheduler = ASHAScheduler( time_attr            = 'training_iteration',
                               max_t                = cfg.model.n_epochs,
                               grace_period         = 1,
                               reduction_factor     = 2)
    
    tuner     = tune.Tuner( tune.with_resources(tune.with_parameters(_hyperparameter_tune, cfg = cfg), resources ={"cpu": 2, "gpu": 1/3}),
                                tune_config =   tune.TuneConfig( metric         = "loss",
                                                                 mode           = "min",
                                                                 scheduler      = scheduler,
                                                                 num_samples    = num_samples,
                                                                 reuse_actors   = False),
                                run_config  =   air.RunConfig  ( log_to_file    = True,
                                                                 failure_config = air.FailureConfig(fail_fast=True)),
                                param_space =   config,)
    
    results     = tuner.fit() 
    best_result = results.get_best_result("loss", "min")
    
    print(f"\n\n")
    print(f"INFO: [hyperparameter_tune] Best trial config: {best_result.config}")
    print(f"INFO: [hyperparameter_tune] Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"INFO: [hyperparameter_tune] Best trial final validation accuracy: {best_result.metrics['accuracy']}")