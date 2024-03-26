## hydra imports
import hydra
from omegaconf import DictConfig

## modules
import srcs.Trainer, srcs.Param_Tuner

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def mission(cfg: DictConfig):
    if cfg.mission == 'train':
        srcs.Trainer.train_FX(cfg)
    elif cfg.mission == 'hyperparameter_tune':
        srcs.Param_Tuner.run(cfg) 

if __name__ == "__main__":
    mission()


