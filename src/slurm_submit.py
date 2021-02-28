# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import sys
import uuid
from argparse import Namespace
from pathlib import Path

import submitit

import train

WORK_DIR = str(Path(__file__).parent.absolute())

def parse_args():
    train_parser = train.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for detection", parents=[train_parser])
    parser.add_argument("--ngpus", default=1, type=int,
                        help="Number of gpus to request on each node")
    parser.add_argument("--vram", default="12GB", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--mem_per_gpu", default=20, type=int)
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=60, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str,
                        help="Job dir. Leave empty for automatic.")
    parser.add_argument("--cluster", default=None, type=str,
                        help="Use to run jobs locally.")
    parser.add_argument("--slurm_partition", default="NORMAL", type=str,
                        help="Partition. Leave empty for automatic.")
    parser.add_argument("--slurm_constraint", default="", type=str,
                        help="Constraint. Leave empty for automatic.")
    parser.add_argument("--slurm_comment", default="", type=str)
    parser.add_argument("--slurm_gres", default="", type=str)
    parser.add_argument("--slurm_exclude", default="", type=str)
    parser.add_argument("--checkpoint_name", default="last.ckpt", type=str)
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/storage/slurm").is_dir():
        path = Path(f"/storage/slurm/{user}/runs")
        path.mkdir(exist_ok=True)
        return path
    raise RuntimeError("No shared folder available")


def get_init_file() -> Path:
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    def __call__(self) -> None:
        sys.path.append(WORK_DIR)

        import train
        self._setup_gpu_args()
        train.main(self.args)

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        import os

        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, self.args.checkpoint_name)
        if os.path.exists(checkpoint_file):
            self.args.resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self) -> None:
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        print(self.args.output_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    # Note that the folder will depend on the job_id, to easily track experiments
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(
        folder=args.job_dir, cluster=args.cluster, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.num_gpus
    nodes = args.nodes
    timeout_min = args.timeout

    if args.slurm_gres:
        slurm_gres = args.slurm_gres
    else:
        slurm_gres = f'gpu:{num_gpus_per_node},VRAM:{args.vram}'

    executor.update_parameters(
        mem_gb=args.mem_per_gpu * num_gpus_per_node,
        # gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=2,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=args.slurm_partition,
        slurm_constraint=args.slurm_constraint,
        slurm_comment=args.slurm_comment,
        slurm_exclude=args.slurm_exclude,
        slurm_gres=slurm_gres
    )

    executor.update_parameters(name="fair_track")

    args.dist_url = get_init_file().as_uri()
    # args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

    if args.cluster == 'debug':
        job.wait()


if __name__ == "__main__":
    main()
