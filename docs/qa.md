# Q & A

1. How to train my customized network using the framework of APS?

    There are two options. The first one is straightforward and just to modify the source code of the APS following the steps:

    * Give your implementation of network structure if needed in `aps.asr`, `aps.sse` or `aps.xxx` (may coming in the future) and decorate it using `@ApsRegisters.xxx.register(...)`.
    * Give your implementation of `Task` if needed in `aps.task` and also decorate it using `@ApsRegisters.task.register(...)`.
    * Give your implementation of dataloader if needed in `aps.loader` and also decorate with `@ApsRegisters.loader.register(...)`.
    * Prepare your training & validation & test data & configuration files and train the models using the scripts [scripts/*.sh](../scripts).

        *If new python files are added, remember to update the `ApsModules` class in `aps/libs.py` to make sure your implementation can be imported correctly*.

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
