import torch
import lightning as L
import hydra


def run_trainer(cfg):
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=cfg.logger,
        callbacks=cfg.callbacks,
        **cfg.trainer
    )
    
    model = CANav(cfg)
    trainer.fit(model)



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg):
    run_trainer(cfg)


if __name__ == "__main__":
    main()