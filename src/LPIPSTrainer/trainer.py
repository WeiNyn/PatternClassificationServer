from src.data.data_module import PatternDataModule
from src.model.model import PatternModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger

import lpips

class Setting:
    model=dict(
        net='alex',
        version='0.1',
        lr=0.05,
        step_size=30,
        gamma=0.1,
        beta=0.5
    )
    data_module=dict(
        folder='/data/HuyNguyen/Demo/PatternClassification/120_samples_database_cut/',
        train_batch_size=64,
        test_batch_size=64,
        train_len=1000,
        test_len=240,
        random_size=(150, 512),
        resize_size=256
    )
    checkpoint = dict(
        save_top_k=3,
        monitor='val_loss'
    )
    logger = dict(
        name='pattern'
    )
    early_stopping = dict(
        enable=True,
        monitor='val_loss',
        patience=10
    )
    trainer=dict(
        max_epochs=10000,
        gpus=1,
        profiler='simple'
    )

def train(setting: Setting):
    model_config = setting.model
    pattern_config = {k: v for k, v in model_config.items() if k not in ['net', 'version']}
    
    model = PatternModel(model=lpips.LPIPS(net=model_config['net'],
                                           version=model_config['version']),
                         **pattern_config)
    
    
    data_module_config = setting.data_module
    data_module = PatternDataModule(**data_module_config)
    
    checkpoint_config = setting.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        verbose=True,
        mode='min',
        **checkpoint_config
    )
    
    logger = TensorBoardLogger('lightning_logs', name=setting.logger['name'])
    
    early_stopping_config = setting.early_stopping
    es_enable = early_stopping_config['enable']
    if es_enable is True:
        early_stopping_callback = EarlyStopping(**{k: v for k, v in early_stopping_config.items() if k != 'enable'})
    
    
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback] if es_enable is True else None,
        progress_bar_refresh_rate=30,
        **setting.trainer
    )
    
    trainer.fit(model, data_module)
    
    
if __name__ == '__main__':
    train(Setting())