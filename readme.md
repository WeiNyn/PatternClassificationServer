# Pattern Classification Server

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The source code for pattern classification server, including:

- Route for demo-pipeline
- Route for training model
- Source code for testing, training model indepedent on server

## Requirements

- FastAPI
- Uvicorn
- Torch
- Torch-lightning
- LPIPS
- Torchvision

## Setting

All the server setting stored in src/setting.py
Please eddit the setting file if you don't really need to modify the way the server works.
```python
class Setting:
    device: str = 'cuda'
    sample_dir: str = '/data/part_detect_smart_factory/test_pattern_new_120/120_samples_database_cut/' #path to folder that containts samples for classification and for training (ImageFolder format)
    model_path: str = 'model/pl_model_0517.pth' #path to model
    max_image_size: int = 256 #the maximum size of image that will feed into model
    
    run_train: bool = Fasle #wheather or not re-train the model on the give sample_dir
    
    model=dict(
        net='alex',
        version='0.1',
        lr=0.05,
        step_size=5,
        gamma=0.1,
        beta=0.5
    ) #params for training model
    
    data_module=dict(
        folder='/data/HuyNguyen/Demo/PatternClassification/120_samples_database_cut/',
        train_batch_size=64,
        test_batch_size=64,
        train_len=1000,
        test_len=200,
        random_size=(150, 512),
        resize_size=256
    ) #params for datamodule (read the pytorch-lightning datamodule for Trainer)
    
    checkpoint = dict(
        save_top_k=1,
        monitor='val_loss'
    ) #parasm for checkpoint callback
    
    logger = dict(
        name='pattern'
    ) #params for logger callback
    
    early_stopping = dict(
        enable=True,
        monitor='val_loss',
        patience=3
    ) #params for early stoping callback
    
    trainer=dict(
        max_epochs=10,
        gpus=1,
        profiler=None
    ) #params for training
```
## Run server

Please install all requirements before running server

```sh
uvicorn main:app --port 8888 --host 0.0.0.0
```

Check the live document for API at: http://localhost:8888/docs


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
