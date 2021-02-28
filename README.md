# DVL Slurm Submit Tutorial

This repository demonstrates the submission of Slurm jobs with an `*.sbatch` file or the Submitit Python tool. To this end, we provide a [PyTorch Lighting](https://github.com/PyTorchLightning/pytorch-lightning) MNIST example.

In order to submit the example in `src/train.py` follow:
1. Install Python packages with `pip3 install -r requirements.txt`
2. Submit to Slurm with `sbatch slurm_submit.sbatch`

The submission configuration is located on the top of an `*.sbatch` script. Further configuration options can be found [here](https://slurm.schedmd.com/sbatch.html).

## Multi-GPU training with Submitit

The [Submitit](https://github.com/facebookincubator/submitit) tool provides a Python interface for Slurm job submissions and facilitates particularly the submission of multi-gpu jobs. The example of this tutorial can be submitted with the following command:

```
python src/slurm_submit.py \
    --ngpus 1 \
    --cluster slurm \
    --output_dir logs/mnist_example
```

Setting `--cluster debug` starts the job locally and allows for testing/debugging of your code. The log files are written under `/storage/slurm/{USER}/runs`. Submitit creates `*.sbatch` scripts, log and error files for each GPU.

## Preemption

> Slurm supports job preemption, the act of "stopping" one or more "low-priority" jobs to let a "high-priority" job run. When a job that can preempt others has allocated resources that are already allocated to one or more jobs that could be preempted by the first job, the preemptable job(s) are preempted.
> [Offcial Slurm webpage](https://slurm.schedmd.com/preempt.html)

The PyTorch Lighting framework already handles most of the bookkeeping (model saving, logging, and resuming) on its own. However, for this example we very briefly demonstrate how a codebase can be adapted to check for an existing model and reload it upon preemption, i.e., restart.

In order to manually trigger a preemption execute: `scontrol requeue job_id`. The job will get requeued and once it restarts should resume training. You can test preemption and make yourself familiar with the example code provided with this tutorial.

Each project has a unique structure with different frameworks, visualizations (Visdom vs. TensorBoard), ways of loading configuration parameters. Therefore it is required for you do adapt your code accordingly. For a full resumption of your training you should consider the following:

1. Model state
2. Optimizer and scheduler states
3. Number of epochs
4. Visualization

We might add examples for specific resumption scenarios.