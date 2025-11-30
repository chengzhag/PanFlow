import time
import datetime
from concurrent.futures import ProcessPoolExecutor
from settings import args, get_logger, DEBUG, SCHEDULER
from process_video import process_video
import argparse
import contextlib
import os
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.panel import Panel
from collections import OrderedDict
from tqdm.auto import tqdm
from functools import partial


args |= {
    "processes": 0 if DEBUG else 64,
}


def init_curation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    cli_args = parser.parse_args()

    log_path = args["output_root"] / "logs" / f"{SCHEDULER}-{'-'.join(cli_args.steps)}.txt"
    logger = get_logger(__name__, log_path, False)
    logger.info(f"Running steps: {cli_args.steps}")
    logger.info(f"Scheduler: {SCHEDULER}")

    video_root = args["data_root"] / "videos"
    videos = list(video_root.glob("*.mp4"))
    videos.sort()
    if cli_args.samples is not None:
        videos = videos[:cli_args.samples]
    if DEBUG:
        videos = videos[:200]
    logger.info(f"Found {len(videos)} videos")

    return cli_args, videos, logger


def worker_scan_video(video, steps):
    with open(os.devnull, 'w') as fnull, \
            contextlib.redirect_stdout(fnull), \
            contextlib.redirect_stderr(fnull):
        if not process_video(video, steps, return_completion=True):
            return video
    return None


def scan_completion(videos, steps, logger):
    worker = partial(worker_scan_video, steps=steps)

    if DEBUG:
        results = []
        for video in tqdm(videos, desc="Scanning for completed videos"):
            results.append(worker(video))
    else:
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(
                executor.map(worker, videos, chunksize=16),
                total=len(videos),
                desc="Scanning for completed videos",
            ))
    resume_videos = [video for video in results if video is not None]

    logger.info(f"{len(resume_videos)} / {len(videos)} videos to be processed")
    return resume_videos


def curate_video(video, silent=True, steps=None, force=False, logger=None):
    start_time = time.time()
    video_id = video.stem
    # logger.info(f"Processing video: {video}")

    try:
        if DEBUG:
            time.sleep(0.1)
        elif silent:
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                process_video(video, steps, force=force)
        else:
            process_video(video, steps, force=force)
        elapsed_time = time.time() - start_time
        elapsed_time = str(datetime.timedelta(seconds=int(elapsed_time)))
        info = f"{video_id} processed in {elapsed_time}"
    except Exception as e:
        info = f"Error processing {video_id}: {e}"
        log_path = args["output_root"] / "logs" / f"{video_id}.txt"
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
                info += f"""
Log file: {log_path}
Last 10 lines of the log file:
{''.join(lines[-10:])}
"""
        else:
            info += f"\nLog file not found: {log_path}"
    if logger is not None:
        logger.info(info)
    return info


def track_progress(futures, logger, flag_folder=None):
    console = Console()

    layout = Layout()
    layout.split(
        Layout(name="upper", minimum_size=3),
        Layout(name="lower", minimum_size=3),
        Layout(name="progress", size=1)
    )
    layout["lower"].split_row(
        Layout(name="lower_left"),
        Layout(name="lower_right")
    )

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        expand=True
    )
    progress_task = progress.add_task("Processing videos", total=len(futures))

    running_tasks = OrderedDict()
    completed_tasks = OrderedDict()

    with Live(layout, console=console, screen=True, refresh_per_second=4):
        while futures:
            if flag_folder is not None:
                running_videos = os.listdir(flag_folder)
            curr_time = time.time()
            with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                reamining_videos = []
                for future, video_id in list(futures.items()):
                    running = future.running() if flag_folder is None else video_id in running_videos
                    if running and video_id not in running_tasks:
                        running_tasks[video_id] = curr_time
                    elif future.done():
                        logger.info(future.result())
                        completed_tasks[video_id] = curr_time - running_tasks.pop(video_id, curr_time)
                        del futures[future]
                        progress.update(progress_task, advance=1)
                    else:
                        reamining_videos.append(video_id)

            completed_infos = [
                f"{vid}\xa0{datetime.timedelta(seconds=int(st))}"
                for vid, st in list(reversed(completed_tasks.items()))[:500]
            ]
            completed_infos = "\t".join(completed_infos)

            running_infos = [
                f"{vid}\xa0{datetime.timedelta(seconds=int(curr_time - st))}"
                for vid, st in running_tasks.items()
            ]
            running_infos = "\t".join(running_infos)

            remaining_infos = reamining_videos[:200]
            remaining_infos = "\t".join(remaining_infos)

            layout["upper"].update(Panel(running_infos, title=f"{len(running_tasks)} Running Tasks"))
            layout["lower_left"].update(Panel(completed_infos, title=f"{len(completed_tasks)} Completed Tasks"))
            layout["lower_right"].update(Panel(remaining_infos, title=f"{len(futures)} Remaining Tasks"))
            layout["progress"].update(progress)

            time.sleep(0.2)


def main():
    cli_args, videos, logger = init_curation()

    if not cli_args.force:
        videos = scan_completion(videos, cli_args.steps, logger=logger)
    if not videos:
        return

    logger.info(f"Processes: {args['processes']}")
    if args["processes"] == 0:
        for video in tqdm(videos, desc="Processing videos"):
            curate_video(
                video,
                silent=False,
                steps=cli_args.steps,
                force=cli_args.force,
                logger=logger,
            )
    else:
        with ProcessPoolExecutor(max_workers=args["processes"]) as executor:
            futures = {executor.submit(
                curate_video,
                video,
                silent=True,
                steps=cli_args.steps,
                force=cli_args.force,
            ): video.stem for video in videos}

            track_progress(futures, logger)


if __name__ == "__main__":
    main()
