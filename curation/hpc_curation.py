import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider, PBSProProvider, LocalProvider
from parsl.launchers import SrunLauncher, SingleNodeLauncher
from parsl.addresses import address_by_hostname
from settings import DEBUG, SCHEDULER, args
from batch_curation import init_curation, scan_completion, track_progress
import shutil


template_string = """#!/bin/bash

#PBS -S /bin/bash
#PBS -N ${jobname}
#PBS -m n
#PBS -l walltime=$walltime
#PBS -l ncpus=${ncpus}
#PBS -o ${job_stdout_path}
#PBS -e ${job_stderr_path}
${scheduler_options}

${worker_init}

export JOBNAME="${jobname}"

${user_script}

"""


def main():
    cli_args, videos, logger = init_curation()
    steps = cli_args.steps
    force = cli_args.force
    task_type = "gpu" if steps is None or "watermark_score" in steps else "cpu"

    if not force:
        videos = scan_completion(videos, steps, logger=logger)
    if not videos:
        return

    # settings
    # if DEBUG:
    #     parsl.set_stream_logger(level=LOGGING_LEVEL)
    if task_type == "cpu":
        mem_per_worker = 2
        if SCHEDULER == "local":
            max_workers_per_node = 1 if DEBUG else 16
            max_blocks = 1
            cores_per_worker = 1
        else:
            max_workers_per_node = 1 if DEBUG else 32
            max_blocks = 1 if DEBUG else 24
            cores_per_worker = 0.5
        cores_per_node = max(int(max_workers_per_node * cores_per_worker), 1)
    else:
        mem_per_worker = 1
        if SCHEDULER == "local":
            max_workers_per_node = 1 if DEBUG else 32
            max_blocks = 1
            cores_per_worker = 1
        else:
            max_workers_per_node = 1 if DEBUG else 128
            max_blocks = 1 if DEBUG else 4
            cores_per_worker = 0.25
        cores_per_node = max(int(max_workers_per_node * cores_per_worker), 1)
    max_blocks = min(max_blocks, len(videos) // max_workers_per_node)

    logger.info(f"Task type: {task_type}")
    logger.info(f"Max workers per node: {max_workers_per_node}")
    logger.info(f"Max blocks: {max_blocks}")

    if SCHEDULER == "slurm":
        worker_init = """
source ~/.bashrc
module load cuda/11.7
conda activate pano_video_curation
export LD_LIBRARY_PATH=${HOME}/usr/local/lib64:${HOME}/usr/local/lib:$LD_LIBRARY_PATH
export PATH=${HOME}/usr/local/bin:$PATH
"""
        if task_type == "cpu":
            def cpu_provider(partition, qos, account):
                return SlurmProvider(
                    partition=partition,
                    mem_per_node=mem_per_worker * max_workers_per_node,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cores_per_node=cores_per_node,
                    qos=qos,
                    account=account,
                    worker_init=worker_init,
                    launcher=SrunLauncher(),
                    walltime="168:00:00",
                )

            providers = {}
            for partition, qos, account in [
                ["fitc", "fitcq", "dv94"],
                # ["comp", None, "dv90"],
                # ["comp", None, "dv94"],
                # ["comp", None, "dv91"],
            ]:
                providers[f"{partition}_{account}"] = cpu_provider(partition, qos, account)
        else:
            def gpu_provider(partition, qos, account):
                return SlurmProvider(
                    partition=partition,
                    mem_per_node=mem_per_worker * max_workers_per_node,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cores_per_node=cores_per_node,
                    qos=qos,
                    account=account,
                    worker_init=worker_init,
                    launcher=SrunLauncher(),
                    walltime="24:00:00",
                    scheduler_options="#SBATCH --gres=gpu:1",
                )

            providers = {}
            for partition, qos, account in [
                # ["fit", "fitq", "dv90"],
                ["fit", "fitq", "dv94"],
                # ["gpu", None, "dv90"],
                # ["gpu", None, "dv94"],
                # ["gpu", None, "dv91"],
                # ["desktop", "desktopq", "dv90"],
                # ["desktop", "desktopq", "dv94"],
                # ["desktop", "desktopq", "dv91"],
            ]:
                providers[f"{partition}_{account}"] = gpu_provider(partition, qos, account)
    elif SCHEDULER == "pbs":
        worker_init = """
source ~/.bashrc
module load cuda/11.7.0
conda activate pano_video_curation
export LD_LIBRARY_PATH=${HOME}/usr/local/lib64:${HOME}/usr/local/lib:$LD_LIBRARY_PATH
export PATH=${HOME}/usr/local/bin:$PATH
"""
        if task_type == "cpu":
            def cpu_provider(queue):
                return PBSProProvider(
                    queue=queue,
                    account="zv92",
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cpus_per_node=cores_per_node,
                    worker_init=worker_init,
                    scheduler_options=f"#PBS -l mem={mem_per_worker * max_workers_per_node}GB,jobfs=200GB,storage=gdata/zv92,wd",
                    launcher=SingleNodeLauncher(),
                    walltime="48:00:00",
                )

            providers = {}
            for queue in [
                "normal",
                # "normalbw",
                # "normalsr",
            ]:
                providers[queue] = cpu_provider(queue)
        else:
            providers = {
                "gpu": PBSProProvider(
                    queue="gpuvolta",
                    account="zv92",
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cpus_per_node=cores_per_node,
                    worker_init=worker_init,
                    scheduler_options=f"#PBS -l mem={mem_per_worker * max_workers_per_node}GB,ngpus=1,jobfs=200GB,storage=gdata/zv92,wd",
                    launcher=SingleNodeLauncher(),
                    walltime="48:00:00",
                )
            }
        for provider in providers.values():
            provider.template_string = template_string
    else:
        worker_init = """
source ~/.bashrc
conda activate pano_video_curation
"""
        providers = {
            "local": LocalProvider(
                init_blocks=1,
                min_blocks=0,
                max_blocks=max_blocks,
                worker_init=worker_init,
            )
        }

    executors = []
    for label, provider in providers.items():
        executors.append(
            HighThroughputExecutor(
                label=label,
                worker_debug=DEBUG,
                cores_per_worker=cores_per_worker,
                mem_per_worker=mem_per_worker,
                max_workers_per_node=max_workers_per_node,
                address=address_by_hostname(),
                provider=provider,
            )
        )

    config = Config(
        executors=executors,
        strategy="htex_auto_scale",
        run_dir=str(args["debug_root"] / "runinfo"),
        retries=3,
    )
    parsl.load(config)

    flag_folder = args["output_root"] / "processing"
    shutil.rmtree(flag_folder, ignore_errors=True)
    flag_folder.mkdir(parents=True, exist_ok=True)

    @parsl.python_app(executors=list(providers.keys()))
    def run_task(video):
        import sys
        import os
        sys.path.append(".")
        from batch_curation import curate_video
        processing_flag = flag_folder / video.stem
        processing_flag.touch()
        result = curate_video(
            video,
            steps=steps,
            force=force,
        )
        os.remove(processing_flag)
        return result

    try:
        futures = {run_task(video): video.stem for video in videos}
        track_progress(futures, logger, flag_folder)
    except Exception as e:
        raise e
    finally:
        logger.info("Cancelling all futures...")
        for future in futures:
            if not future.done():
                try:
                    future.cancel()
                except NotImplementedError:
                    pass
        logger.info("All futures cancelled.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        parsl.clear()
