class Setting:
    device: str = 'cuda'
    sample_dir: str = '/data/part_detect_smart_factory/test_pattern_new_120/120_samples_database_cut/'
    model_path: str = 'model/pl_model_0517.pth'
    max_image_size: int = 256
    
    run_train: bool = False
    
    model=dict(
        net='alex',
        version='0.1',
        lr=0.05,
        step_size=5,
        gamma=0.1,
        beta=0.5
    )
    
    data_module=dict(
        folder='/data/HuyNguyen/Demo/PatternClassification/120_samples_database_cut/',
        train_batch_size=64,
        test_batch_size=64,
        train_len=1000,
        test_len=200,
        random_size=(150, 512),
        resize_size=256
    )
    
    checkpoint = dict(
        save_top_k=1,
        monitor='val_loss'
    )
    
    logger = dict(
        name='pattern'
    )
    
    early_stopping = dict(
        enable=True,
        monitor='val_loss',
        patience=3
    )
    
    trainer=dict(
        max_epochs=10,
        gpus=1,
        profiler=None
    )