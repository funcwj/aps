# Q & A

1. How to train my customized network under the framework of aps?

    There are two options. The first one is to modify the code of the aps directly following the steps:

    * Give your implementation of network structure if needed in `aps.asr`, `aps.sse` or `aps.xxx` (may coming in the future) and register it in `asr_nnet_cls` or `sse_nnet_cls`.
    * Give your implementation of `Task` if needed in `aps.task` and also register it in `task_cls`.
    * Give your implementation of dataloader if needed in `aps.loader` and also register it in `loader_cls`.
    * Prepare your training & validation & test data & configuration files and train the models using the scripts [scripts/*.sh](../scripts).

    Another way only requires us to modify the training configurations. Assuming we have `my_nnet.py`, `my_task.py` under `/path/to/my_code`, the `.yaml` configuration like
    ```yaml
    nnet: /path/to/my_code/my_nnet.py:MyNnet
    nnet_conf:
        # put parameters here
        ...
    task: /path/to/my_code/my_task.py:MyTask
    task_conf:
        # put parameters here
        ...
    # other configurations
    ...
    ```
    could be used. In this case, *please remember to make sure there is no import errors in your python code*.
