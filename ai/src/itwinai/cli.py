"""
Command line interface for out Python application.
You can call commands from the command line.
Example

>>> $ itwinai --help

"""

# NOTE: import libs in the command"s function, not here.
# Otherwise this will slow the whole CLI.

# from typing import Optional
import os
import typer


app = typer.Typer()


@app.command()
def train(
    input: str = typer.Option(
        "unk",
        help="Path to training dataset."
    ),
    config: str = typer.Option(
        "unk",
        help="Path to training configuration file."
    ),
    output: str = typer.Option(
        "logs/",
        help="Path to logs storage."
    )
):
    """
    Train a neural network defined as a Pytorch Lightning model.
    """
    import copy
    import mlflow
    from lightning.pytorch.cli import LightningCLI

    from itwinai.utils import load_yaml, flatten_dict
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )

    os.makedirs(output, exist_ok=True)
    train_config = load_yaml(config)

    log_conf = train_config['logger']
    mlflow.set_tracking_uri("file:" + output)
    mlflow.set_experiment(log_conf['experiment_name'])
    mlflow.pytorch.autolog(
        log_every_n_epoch=log_conf['log_every_n_epoch'],
        log_every_n_step=log_conf['log_every_n_steps'],
        registered_model_name=log_conf['registered_model_name']
    )

    # Note: we use autolog and MlFlowLogger combined:
    # - MlFlow logger provides better flexibility
    # - autolog takes care of repetitive operations
    # Ref: https://github.com/Lightning-AI/lightning/discussions/11197

    # Load training configuration
    lightning_conf = load_yaml(train_config['train']['path'])

    # Start Mlflow run
    with mlflow.start_run(description=log_conf['description']):
        # Log hyperparameters
        config_params = copy.copy(train_config)
        config_params['cli.input'] = input
        config_params['cli.output'] = output
        config_params['cli.config'] = config
        mlflow.log_params(flatten_dict(config_params))

        # Save config file used for this specific training run
        # for reproducibility
        mlflow.log_artifact(config)

        # Update lightning MLFlow logger constructor args
        # Infer MlFlow conf from pre-configured mlflow client
        lightning_conf['trainer']['logger']['init_args'].update(dict(
            experiment_name=mlflow.get_experiment(
                mlflow.active_run().info.experiment_id
            ).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model='all',
            run_id=mlflow.active_run().info.run_id
        ))

        cli = LightningCLI(
            args=lightning_conf,
            model_class=ItwinaiBasePlModule,
            datamodule_class=ItwinaiBasePlDataModule,
            run=False,
            save_config_kwargs={"overwrite": True,
                                "config_filename": "pl-training.yml"},
            subclass_mode_model=True,
            subclass_mode_data=True
        )

        # Train + validation, and test
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        cli.trainer.test(dataloaders=cli.datamodule, datamodule=cli.datamodule)

        # Save updated lightning conf as an mlflow artifact
        mlflow.log_artifact(
            os.path.join(cli.trainer.log_dir, "pl-training.yml")
        )


@app.command()
def predict(
    config: str = typer.Option(
        "unk",
        help="Path to inference configuration file."
    ),
    output: str = typer.Option(
        "preds/",
        help="Path to predictions storage."
    )
):
    """
    Apply a pre-trained neural network to a set of unseen data.
    """
    import mlflow
    from lightning.pytorch.cli import LightningCLI
    from lightning.pytorch.trainer.trainer import Trainer
    import torch

    from itwinai.utils import load_yaml
    from itwinai.plmodels.base import (
        ItwinaiBasePlModule,
        ItwinaiBasePlDataModule
    )

    # Load ML configuration
    ml_conf = load_yaml(config)
    ml_conf = ml_conf['inference']

    os.makedirs(output, exist_ok=True)

    mlflow.set_tracking_uri(ml_conf['tracking_uri'])

    # Download training configuration
    train_conf_path = mlflow.artifacts.download_artifacts(
        run_id=ml_conf['run_id'],
        artifact_path=ml_conf['train_config_artifact_path'],
        dst_path='tmp/',
        tracking_uri=mlflow.get_tracking_uri()
    )

    # Download last ckpt
    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=ml_conf['run_id'],
        artifact_path=ml_conf['ckpt_path'],
        dst_path='tmp/',
        tracking_uri=mlflow.get_tracking_uri()
    )

    # Instantiate PL model
    lightning_conf = load_yaml(train_conf_path)

    cli = LightningCLI(
        args=lightning_conf,
        model_class=ItwinaiBasePlModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None
    )

    # Load best model
    loaded_model = cli.model.load_from_checkpoint(
        ckpt_path,
        lightning_conf['model']['init_args']
    )

    # Load Data module
    if ml_conf.get('data') is not None:
        # New/updated datamodule
        loaded_data_module: ItwinaiBasePlDataModule = None
        raise NotImplementedError
    else:
        # Reuse same datamodule used for training
        loaded_data_module: ItwinaiBasePlDataModule = cli.datamodule

    # Test best model once again (TODO: remove)
    trainer = Trainer()
    trainer.test(
        loaded_model,
        dataloaders=loaded_data_module,
        datamodule=loaded_data_module
    )  # , ckpt_path='best')

    # Predict
    predictions = trainer.predict(
        loaded_model,
        datamodule=loaded_data_module
    )  # , ckpt_path='best')
    pred_class_names = loaded_data_module.preds_to_names(
        torch.cat(predictions)
    )

    # Save list of predictions as class names
    with open(os.path.join(output, 'predictions.txt'), 'w') as preds_file:
        preds_file.write('\n'.join(pred_class_names))


@app.command()
def visualize(
    path: str = typer.Option(
        "logs/",
        help="Path to logs storage."
    ),
):
    """
    Visualize Mlflow logs.
    """
    import subprocess
    subprocess.run(f"mlflow ui --backend-store-uri {path}".split())


if __name__ == "__main__":
    app()
