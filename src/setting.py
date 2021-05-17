class Setting:
    device: str = 'cuda'
    # sample_dir: str = 'image_120_wider_sub/'
    sample_dir: str = '/data/part_detect_smart_factory/test_pattern_new_120/120_samples_database_cut/'
    model: str = 'model/pl_model_0517.pth'
    # model: str = None
    max_image_size: int = 256