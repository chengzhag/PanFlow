import cv2
import imageio
import numpy as np
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
from settings import DEBUG, SCHEDULER, args, get_logger
import json
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
import yaml
import subprocess
import tempfile
from pathlib import Path
import re
from contextlib import ExitStack
from watermark_laion import run_watermark_laion
from PIL import Image
import os
from transitions import Machine
from typing import Union, List
from decord import VideoReader, cpu
import ffmpeg


args |= {
    "output_root": args["data_root"] if not DEBUG else args["debug_root"],
    "video_check": {
        "sample_frames": 5,
    },
    "format_check": {
        "sample_frames": 5,
        "fisheye_thres": 0.02,
    },
    "detect_scenes": {
        "min_scene_len": 5.,
        "scene_cut_len": 0.,
    },
    "slam_clips": {
        "min_clip_len": 3.,
        "max_clip_len": 10.,
        "clip_cut_len": 1.,
        "trans_len": 2.,
        "slam_config": {
            # "Mapping": {
            #     "baseline_dist_thr_ratio": 0.2,  # enlarged from 0.02 for more strict keyframe insertion and triangulation
            # },
            "KeyframeInserter": {
                "max_interval": 10.,  # enlarged from 1.0 for more strict keyframe insertion
                # "lms_ratio_thr_almost_all_lms_are_tracked": 0.8,  # lowered from 0.9 for more strict keyframe insertion
                # "lms_ratio_thr_view_changed": 0.7,  # lowered from 0.8 for more strict keyframe insertion
            },
            # "Initializer": {
            #     "parallax_deg_threshold": 3.,  # enlarged from 1.0 for more strict initialization
            # },
            "Tracking": {
                "enable_auto_relocalization": False,  # disabled for transition detection
            },
        }
    },
    "motion_score": {
        "speed_up": 10,
        "min_size": 256,
    },
    "watermark_score": {
        "sample_frames": 5,
    },
}


def load_meta(
    meta_path,
):
    with open(meta_path, "r") as f:
        return json.load(f)


def video_check(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if not video_valid(video_path, logger):
        logger.info(f"Video {video_path} is invalid")
        func_meta |= func_args | {
            "pass": False,
            "reason": "Invalid video file",
        }
        return

    # read the video
    cap = cv2.VideoCapture(str(video_path))

    # load a few frames and check if they are valid
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = total_frames // (func_args["sample_frames"] - 1)
    idx_frames = [int(i) for i in range(0, total_frames, step_frames)]
    for idx in idx_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if frame is None:
            logger.info(f"Frame {idx} is None")
            func_meta |= func_args | {
                "pass": False,
                "reason": "Incomplete video file",
            }
            return

    func_meta |= func_args


def format_check(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    # read the video
    cap = cv2.VideoCapture(str(video_path))

    # load a few frames and check the format
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = total_frames // (func_args["sample_frames"] - 1)
    idx_frames = [int(i) for i in range(0, total_frames, step_frames)]
    frames = []
    for idx in idx_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)

    # check if left is fisheye by masking with a circle
    fisheye_mask = np.ones((256, 256), dtype=np.uint8) * 255
    cv2.circle(fisheye_mask, (128, 128), 128, 0, -1)
    if DEBUG:
        cv2.imwrite(str(output_dir / "mask.png"), fisheye_mask)
    fisheye_mask = fisheye_mask.astype(bool)

    # check if the video is 3D format
    # psnr_list = []
    ssim_list = []
    probs_fisheye = []
    boarder_vars = []
    for idx, frame in enumerate(frames):
        # check if left and right are similar with ssim
        width = frame.shape[1]
        left = frame[:, :width//2]
        right = frame[:, width//2:]
        left, right = cv2.resize(left, (256, 256)), cv2.resize(right, (256, 256))

        if DEBUG:
            cv2.imwrite(str(output_dir / f"left_{idx}.png"), left)
            cv2.imwrite(str(output_dir / f"right_{idx}.png"), right)

        # # check with psnr
        # psnr_list.append(psnr(left, right))

        # check with ssim
        left, right = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        ssim_list.append(float(ssim(left, right)))

        # check if left is fisheye by masking with a circle and checking the mean value
        sample_pixels = left[fisheye_mask] / 255
        prob_fisheye = (sample_pixels < func_args["fisheye_thres"]).mean()
        probs_fisheye.append(float(prob_fisheye))

    # psnr_list = np.array(psnr_list)
    # logger.debug(f"PSNR: {psnr_list}")
    logger.debug(f"3D SSIM: {ssim_list}")
    logger.debug(f"Fisheye prob: {probs_fisheye}")

    formats_meta = {
        "scores": {
            "3d": float(np.mean(ssim_list)),
            "fisheye": float(np.mean(probs_fisheye)),
        },
    }

    func_meta |= func_args | formats_meta


def video_montage(
    video_path,
    output_path,
    num_frames=10,
    fps=5,
):
    # read the video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = total_frames // (num_frames - 1)
    idx_frames = [int(i) for i in range(0, total_frames, step_frames)]
    sample_frames = []
    for idx in idx_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (512, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sample_frames.append(frame)

    # create a video montage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, sample_frames, fps=fps, loop=0)


def detect_scenes(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    # detect scenes
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.add_detector(ThresholdDetector())
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # if no scene is detected, add the whole video as a scene
    if not scene_list:
        start = FrameTimecode(0, video.frame_rate)
        end = FrameTimecode(video.frame_number, video.frame_rate)
        scene_list = [(start, end)]

    # filter scenes
    min_scene_frames = func_args["min_scene_len"] + 2 * func_args["scene_cut_len"]
    filtered_scene_list = []
    for start, end in scene_list:
        scene_len = (end - start).get_seconds()
        if scene_len >= min_scene_frames:
            start = start + float(func_args["scene_cut_len"])
            end = end - float(func_args["scene_cut_len"])
            filtered_scene_list.append((start, end))

    if DEBUG and filtered_scene_list:
        from scenedetect import split_video_ffmpeg
        split_video_ffmpeg(
            str(video_path),
            filtered_scene_list,
            str(args["debug_root"] / "detect_scenes" / video_path.stem),
            show_progress=True,
        )

    func_meta |= func_args | {
        "scene_frames": [(int(start.get_frames()), int(end.get_frames())) for start, end in filtered_scene_list],
        "scene_seconds": [(start.get_seconds(), end.get_seconds()) for start, end in filtered_scene_list],
    }
    logger.info(f"Detected {len(func_meta['scene_frames'])} scenes")


def slam_clips(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if "detect_scenes" not in meta:
        logger.warning("No detect_scenes found, skipping")
        return

    with ExitStack() as stack:
        tmp_dir = output_dir if DEBUG else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        config, fps, video_end, slam_cmd = prepare_slam(video_path)

        # save config
        for key, value in func_args["slam_config"].items():
            for k, v in value.items():
                config[key][k] = v
        config_path = tmp_dir / "slam_clips.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # run slam on each scene to detect clips
        logger.info("Detecting clips with SLAM")
        clip_cut_seconds = func_args["clip_cut_len"]
        min_clip_seconds = func_args["min_clip_len"] + 2 * clip_cut_seconds
        min_clip_frames = round(min_clip_seconds * fps)
        max_clip_seconds = func_args["max_clip_len"]
        max_clip_frames = round(max_clip_seconds * fps)
        clip_cut_frames = round(clip_cut_seconds * fps)
        clips = []

        def add_clip(clip_info):
            nonlocal num_scene_clips
            logger.info(f"{clip_info} for Clip {clip_id}")
            clip_meta = {
                "scene_id": scene_id,
                "clip_id": clip_id,
                "clip_name": clip_name,
                "frames": clip_frames,
                "seconds": clip_seconds,
                "info": clip_info,
            }
            clips.append(clip_meta)
            num_scene_clips += 1
            save_meta(clip_meta_path, clip_meta)
            if DEBUG:
                trans_frames = int(func_args["trans_len"] * fps)
                for trans_name, trans_center, save_trans in zip(
                    ["begin", "end"],
                    (clip_frames[0] - clip_cut_frames, clip_frames[1] + clip_cut_frames),
                    has_trans,
                ):
                    if not save_trans:
                        continue
                    trans_video_path = output_dir / f"Transition-{clip_id:03d}-{trans_name}.mp4"
                    logger.info(f"Saving {trans_name} transition to {trans_video_path}")
                    trans_begin = max(0, trans_center - trans_frames // 2)
                    trans_end = min(video_end, trans_center + trans_frames // 2)

                    cut_video(
                        video_path,
                        trans_begin,
                        trans_end,
                        trans_video_path,
                        logger,
                    )

        for scene_id, (scene_frames, scene_seconds) in enumerate(zip(
            meta["detect_scenes"]["scene_frames"],
            meta["detect_scenes"]["scene_seconds"],
        )):
            scene_id += 1
            clip_end = scene_frames[0] - 1
            logger.info(f"Processing scene {scene_id} / {len(meta['detect_scenes']['scene_frames'])}")
            logger.info(f"Scene frames: {scene_frames}")
            logger.info(f"Scene seconds: {scene_seconds}")

            # check if the scene is already processed
            scene_name = f"Scene-{scene_id:03d}"
            scene_meta_path = (output_dir / scene_name).with_suffix(".json")
            scene_processed = scene_meta_path.exists()
            if scene_processed:
                scene_meta = load_meta(scene_meta_path)
            else:
                scene_meta = {
                    "scene_id": scene_id,
                    "frames": scene_frames,
                    "seconds": scene_seconds,
                }

            # use SLAM to detect clips in the scene and generate camera trajectory
            num_scene_clips = 0
            has_trans = [True, True]
            while True:
                has_trans = [has_trans[1], True]

                # check if the clip is already processed
                clip_id = len(clips) + 1
                clip_name = f"Clip-{clip_id:03d}"
                clip_meta_path = (output_dir / clip_name).with_suffix(".json")
                clip_processed = clip_meta_path.exists()
                if clip_processed:
                    logger.info(f"Clip {clip_id} already processed, loading from {clip_meta_path}")
                    clip_meta = load_meta(clip_meta_path)
                    clips.append(clip_meta)
                    num_scene_clips += 1
                    clip_end = clip_meta["frames"][1]

                # initialize the clip
                new_begin = clip_end + 1

                # check if the end of the scene is reached
                if new_begin > scene_frames[1] - min_clip_frames:
                    break

                if clip_processed:
                    continue

                # run mapping to detect the next clip
                begin_seconds = new_begin / fps
                logger.info(f"Running mapping from frame {new_begin} ({begin_seconds:.2f}s) to detect the Clip {clip_id}")
                clip_cmd = slam_cmd + [
                    "--video", str(video_path),
                    "--start-time", str(round((begin_seconds) * 1000)),
                    "--config", str(config_path),
                ]
                curr_max_clip_frames = max_clip_frames + clip_cut_frames
                if has_trans[0]:
                    curr_max_clip_frames += clip_cut_frames
                curr_max_clip_frames = min(
                    curr_max_clip_frames,
                    scene_frames[1] - new_begin + 1,
                )
                mapping = Mapping(
                    clip_cmd,
                    logger,
                    max_frames=curr_max_clip_frames,
                )
                mapping.run()

                # shift the begin and end to the video frames
                clip_begin = new_begin
                clip_end = scene_frames[1] if mapping.end is None else new_begin + mapping.end
                has_trans[1] = clip_end - clip_begin + 1 < curr_max_clip_frames or clip_end >= scene_frames[1]

                # check if the mapping is successful
                if mapping.state == "Failed":
                    clip_end = clip_end + clip_cut_frames
                    logger.info(f"Mapping failed for Clip {clip_id}")
                    continue

                # cut the clip with a margin
                clip_begin = clip_begin + clip_cut_frames if has_trans[0] else clip_begin
                clip_end = clip_end - clip_cut_frames
                clip_begin_seconds = clip_begin / fps
                clip_end_seconds = clip_end / fps
                num_seconds = clip_end_seconds - clip_begin_seconds
                clip_frames = (clip_begin, clip_end)
                clip_seconds = (clip_begin_seconds, clip_end_seconds)
                logger.info(f"Clip frames: {clip_frames}")
                logger.info(f"Clip seconds: {clip_seconds}")

                # check if the clip is too short
                unclip_seconds = num_seconds + 2 * clip_cut_seconds
                if unclip_seconds < min_clip_seconds:
                    clip_end = clip_end + clip_cut_frames * 2
                    logger.info(f"Clip too short, {unclip_seconds} < {min_clip_seconds} seconds")
                    continue

                add_clip(mapping.state)

            if scene_processed:
                if num_scene_clips != scene_meta["num_clips"]:
                    raise Exception(f"Number of clips mismatch ({num_scene_clips} != {scene_meta['num_clips']}) for scene {scene_id}")
                logger.info(f"Scene {scene_id} loaded with {num_scene_clips} clips")
            else:
                scene_meta["num_clips"] = num_scene_clips
                logger.info(f"Scene {scene_id} processed with {num_scene_clips} clips")
                save_meta(scene_meta_path, scene_meta)

    func_meta |= func_args | {
        "clips": clips,
    }
    logger.info(f"Processed {len(clips)} clips")


def export_clips(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if "slam_clips" not in meta:
        logger.warning("No slam_clips found, skipping")
        return

    clips = []
    clip_begins = []
    clip_ends = []
    clip_video_paths = []
    for clip_meta in meta["slam_clips"]["clips"]:
        clip_id = clip_meta["clip_id"]
        clip_name = clip_meta.get("clip_name", f"Clip-{clip_id:03d}")
        clip_begin, clip_end = clip_meta["frames"]
        clip_video_path = (output_dir / clip_name).with_suffix(".mp4")
        clips.append({
            "clip_id": clip_id,
            "video_name": clip_video_path.name,
        })
        clip_begins.append(clip_begin)
        clip_ends.append(clip_end)
        clip_video_paths.append(clip_video_path)

    logger.info(f"Exporting {len(clips)} clips")
    cut_video(
        video_path,
        clip_begins,
        clip_ends,
        clip_video_paths,
        logger,
    )

    func_meta |= func_args | {
        "clips": clips,
    }
    logger.info(f"Exported {len(meta['slam_clips']['clips'])} clips")


def slam_pose(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if "slam_clips" not in meta:
        logger.warning("No slam_clips found, skipping")
        return

    with ExitStack() as stack:
        tmp_dir = output_dir if DEBUG else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        config, _, _, slam_cmd = prepare_slam(video_path)

        # save config
        config_path = tmp_dir / "slam_pose.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        def add_clip(clip_info):
            logger.info(f"{clip_info} for Clip {clip_id}")
            clip_meta = {
                "clip_id": clip_id,
                "info": clip_info,
            }
            clips.append(clip_meta)
            save_meta(clip_meta_path, clip_meta)

        clips = []
        successful_clips = 0
        for clip_meta in meta["slam_clips"]["clips"]:
            clip_id = clip_meta["clip_id"]
            clip_name = clip_meta.get("clip_name", f"Clip-{clip_id:03d}")
            clip_begin, clip_end = clip_meta["frames"]

            # check if the clip is already processed
            clip_meta_path = (output_dir / clip_name).with_suffix(".json")
            if clip_meta_path.exists():
                logger.info(f"Clip {clip_id} already processed, loading from {clip_meta_path}")
                clip_meta = load_meta(clip_meta_path)
                clips.append(clip_meta)
                if clip_meta["info"].startswith("Success"):
                    successful_clips += 1
                continue

            # export the clip
            if not clip_meta["info"].startswith("Success"):
                add_clip(clip_meta["info"])
                continue

            clip_video_path = (args["output_root"] / "export_clips" / video_path.stem / clip_name).with_suffix(".mp4")
            num_frames = clip_end - clip_begin + 1
            if not video_valid(clip_video_path, logger, num_frames):
                clip_video_path = (tmp_dir / clip_name).with_suffix(".mp4")
                cut_video(
                    video_path,
                    clip_begin,
                    clip_end,
                    clip_video_path,
                    logger,
                )

            # run mapping to generate map-db
            logger.info("Running mapping to generate map-db")
            map_db_path = (tmp_dir / clip_name).with_suffix(".msg")
            map_cmd = slam_cmd + [
                "--video", str(clip_video_path),
                "--map-db-out", str(map_db_path),
                "--config", str(config_path),
            ]
            mapping = Mapping(
                map_cmd,
                logger,
            )
            mapping.run()

            # check if mapping is successful, if not, skip the clip
            if mapping.state != "Success" or mapping.end is None or mapping.end != num_frames - 1:
                add_clip("Second mapping failed")
                continue

            # run localization
            with tempfile.TemporaryDirectory() as tmp_traj_dir:
                logger.info("Running localization to generate frame trajectory")
                tmp_traj_dir = Path(tmp_traj_dir)
                loc_cmd = slam_cmd + [
                    "--video", str(clip_video_path),
                    "--disable-mapping",
                    "--temporal-mapping",
                    "--map-db-in", str(map_db_path),
                    "--eval-log-dir", str(tmp_traj_dir),
                    "--config", str(config_path),
                ]
                if run_cmd(loc_cmd, logger):
                    add_clip("Localization failed")
                    continue

                # check if localization is successful, if not, skip the clip
                frame_trajectory = tmp_traj_dir / "frame_trajectory.txt"
                if not frame_trajectory.exists():
                    add_clip("No frame trajectory")
                    continue

                # read the extrinsics
                extrinsics = np.loadtxt(frame_trajectory, dtype=np.float32)
                extrinsics = extrinsics.reshape(-1, 3, 4)
                num_extrinsics = extrinsics.shape[0]
                if num_extrinsics != num_frames:
                    add_clip(f"Trajectory frame not matched ({num_extrinsics} != {num_frames})")
                    continue

            # save the extrinsics
            add_clip("Success")
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save((output_dir / clip_name).with_suffix(".npy"), extrinsics)
            successful_clips += 1

    func_meta |= func_args | {
        "clips": clips,
        "successful_clips": successful_clips,
    }
    logger.info(f"Processed {len(clips)} clips with {successful_clips} success")


def prepare_slam(
    video_path,
):
    # read video information including resolution and fps
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # customize yaml config for video
    with open("equirectangular.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["Camera"]["fps"] = fps
    config["Camera"]["cols"] = width
    config["Camera"]["rows"] = height
    config["Preprocessing"]["min_size"] = height // 4 if DEBUG else height
    if DEBUG:
        config["KeyframeInserter"]["wait_for_local_bundle_adjustment"] = False

    slam_cmd = [
        str(args["stella_vslam_path"]),
        "--vocab", str(args["fbow_path"]),
        "--frame-skip", "1",
        "--no-sleep",
        "--viewer", "socket_publisher" if DEBUG and SCHEDULER == "local" else "none",
        "--auto-term",
        "--log-level", "debug"
    ]

    return config, fps, video_end, slam_cmd


class Mapping:
    states = [
        "Small camera motion",
        "Success",
        "Failed",
    ]

    def __init__(
        self,
        mapping_cmd,
        logger,
        max_frames=None,
    ):
        self.machine = Machine(
            model=self,
            states=Mapping.states,
            initial="Failed",
        )
        self.mapping_cmd = mapping_cmd
        self.logger = logger
        self.max_frames = max_frames

        self.end = None
        self.machine.add_transition("try_to_init_map", "Failed", "Small camera motion")
        self.machine.add_transition("try_to_init_map", "Small camera motion", "Small camera motion")
        self.machine.add_transition("map_created", "Small camera motion", "Success")
        self.machine.add_transition("tracking_lost", "Success", "Success")
        self.machine.add_transition("tracking", "Success", "Success")
        self.machine.add_transition("cholesky_failure", "Success", "Failed")
        self.machine.add_transition("cholesky_failure", "Small camera motion", "Failed")
        self.machine.add_transition("max_frames_reached", "Success", "Success")

    def run(self):
        self.logger.info(f"Running command: {" ".join(self.mapping_cmd)}")
        process = subprocess.Popen(
            self.mapping_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # parse the cli output and monitor mapping progress
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            # if "[I]" in line or "[W]" in line or "[E]" in line:
            self.logger.debug(line)

            # try to initialize with the initial frame and the current frame: frame 279 - frame 798
            if match := re.match(r".*try to initialize with the initial frame and the current frame: frame (\d+) - frame (\d+)", line):
                if int(match.group(1)) > 0:
                    if self.end is None:
                        self.end = int(match.group(1)) - 1
                    break
                self.end = int(match.group(2)) - 1
                self.try_to_init_map()
            # new map created with 147 points: frame 0 - frame 3
            elif match := re.match(r".*new map created with .* points: frame 0 - frame (\d+)", line):
                self.end = int(match.group(1))
                self.map_created()
            # tracking lost: frame 252
            elif match := re.match(r".*tracking lost: frame (\d+)", line):
                self.end = int(match.group(1)) - 1
                self.tracking_lost()
                break
            # tracking succeeded: frame 798
            elif match := re.match(r".*tracking succeeded: frame (\d+)", line):
                self.end = int(match.group(1))
                self.tracking()
            # Cholesky failure, writing debug.txt (Hessian loadable by Octave)
            elif "Cholesky failure, writing debug.txt (Hessian loadable by Octave)" in line:
                self.cholesky_failure()
                break

            if self.end is not None and self.max_frames is not None and self.end >= self.max_frames - 1:
                self.end = self.max_frames - 1
                break

        process.terminate()
        process.wait()

        if self.state == "Failed" and os.path.exists("debug.txt"):
            os.remove("debug.txt")


def run_cmd(
    cmd,
    logger=None,
):
    if logger is not None:
        logger.info(f"Running command: {" ".join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if logger is not None:
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            # if "[I]" in line or "[W]" in line or "[E]" in line:
            logger.debug(line)

    process.wait()
    return process.returncode


def video_valid(
    video_path,
    logger,
    num_frames=None,
):
    if not video_path.exists():
        return False

    check_cmd = [
        "ffmpeg",
        "-v", "error",
        "-xerror",
        "-i", str(video_path),
        "-f", "null",
        "-",
    ]
    if run_cmd(check_cmd, logger):
        return False

    if num_frames is None:
        return True

    cap = cv2.VideoCapture(str(video_path))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != num_frames:
        return False

    return True


def cut_video(
    video_path: Union[str, Path],
    begins: Union[int, List[int]],
    ends: Union[int, List[int]],
    clip_paths: Union[str, Path, List[Union[str, Path]]],
    logger=None,
):
    if not isinstance(begins, list):
        begins = [begins]
    if not isinstance(ends, list):
        ends = [ends]
    if not isinstance(clip_paths, list):
        clip_paths = [clip_paths]
    assert len(begins) == len(ends) == len(clip_paths), "Input lengths must match"

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    height, width, _ = vr[0].shape  # Assume all frames same size

    for i, (begin, end, clip_path) in enumerate(zip(begins, ends, clip_paths)):
        clip_path = Path(clip_path)
        clip_path.parent.mkdir(parents=True, exist_ok=True)

        if logger:
            logger.info(f"[{i+1}/{len(begins)}] Cutting frames {begin}-{end} to {clip_path}")

        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=fps)
            .output(str(clip_path), vcodec='libx264', pix_fmt='yuv420p', crf=23, preset='medium')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        for frame_idx in range(begin, end + 1):
            frame = vr[frame_idx].asnumpy().astype(np.uint8)
            process.stdin.write(frame.tobytes())

        process.stdin.close()

        if logger:
            stderr_output = process.stderr.read().decode('utf-8')
            logger.debug(f"FFmpeg output for clip {clip_path}:\n{stderr_output}")
            logger.info(f"[{i+1}/{len(begins)}] Saved to {clip_path}")

        process.wait()


def motion_score(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if "slam_clips" not in meta:
        logger.warning("No slam_clips found, skipping")
        return

    # https://github.com/huggingface/video-dataset-scripts/tree/main/video_processing
    cap = cv2.VideoCapture(str(video_path))
    resize = (func_args["min_size"] * 2, func_args["min_size"])
    clips = []
    for clip in meta["slam_clips"]["clips"]:
        clip_name = clip["clip_name"]
        clip_meta_path = (output_dir / clip_name).with_suffix(".json")
        if clip_meta_path.exists():
            logger.info(f"Clip already processed, loading from {clip_meta_path}")
            clip_meta = load_meta(clip_meta_path)
            clips.append(clip_meta)
            continue

        begin, end = clip["frames"]
        begin_seconds, end_seconds = clip["seconds"]
        logger.info(f"Processing clip {clip['clip_id']} from {begin} ({begin_seconds:.2f}s) to {end} ({end_seconds:.2f}s)")

        old_frame = None
        farneback = []
        if DEBUG:
            fps = cap.get(cv2.CAP_PROP_FPS)
            vis_flow = cv2.VideoWriter(
                (output_dir / f"Clip-{clip['clip_id']:03d}.mp4").as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                resize,
            )
        for frame_id in range(begin, end, func_args["speed_up"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            hsv = np.zeros((resize[1], resize[0], 3), dtype=frame.dtype)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, resize)
            if old_frame is not None:
                flow_map = cv2.calcOpticalFlowFarneback(
                    old_frame,
                    frame,
                    flow=None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                magnitude, angle = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1])
                if DEBUG:
                    hsv[..., 1] = 255
                    hsv[..., 0] = angle * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    vis_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    vis_flow.write(vis_frame)
                farneback.append(float(np.mean(magnitude)))
            old_frame = frame

        if DEBUG:
            vis_flow.release()

        clips.append({
            "clip_id": clip["clip_id"],
            # "scores": farneback,
            "score": np.mean(farneback),
        })
        logger.info(f"Clip {clip['clip_id']} processed with score {clips[-1]['score']:.2f}")
        save_meta(clip_meta_path, clips[-1])

    func_meta |= func_args | {
        "clips": clips,
    }

    logger.info(f"Processed {len(clips)} clips")


def watermark_score(
    video_path,
    func_args,
    func_meta,
    meta,
    output_dir,
    logger,
):
    if "slam_clips" not in meta:
        logger.warning("No slam_clips found, skipping")
        return

    # read the video
    cap = cv2.VideoCapture(str(video_path))

    # load a few frames and check watermark
    clips = []
    for clip in meta["slam_clips"]["clips"]:
        begin, end = clip["frames"]
        total_frames = end - begin
        step_frames = total_frames // (func_args["sample_frames"] - 1)
        idx_frames = [int(i) for i in range(begin, end, step_frames)]
        frames = []
        for idx in idx_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

        # estimate the watermark score
        scores = run_watermark_laion(frames).tolist()
        logger.debug(f"Watermark scores: {scores}")
        score = np.mean(scores)
        clip_id = clip["clip_id"]
        clips.append({
            "clip_id": clip_id,
            # "scores": scores,
            "score": score,
        })
        logger.info(f"Clip {clip_id} processed with watermark score {score:.2f}")

    func_meta |= func_args | {
        "clips": clips,
    }


def save_meta(
    output_path,
    meta,
):
    # save the meta data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=4)


def process_video(video_path, steps=None, return_completion=False, force=False):
    if not video_path.exists():
        raise Exception(f"Video {video_path} does not exist")

    all_steps = [
        "video_check",
        "format_check",
        "detect_scenes",
        "slam_clips",
        "export_clips",
        "slam_pose",
        "motion_score",
        "watermark_score",
    ]
    if steps is None:
        steps = all_steps
    steps_force = {}
    for step in steps:
        if step.startswith("*"):
            step = step[1:]
            steps_force[step] = True
        else:
            steps_force[step] = False
    steps = steps_force

    video_id = video_path.stem
    if return_completion:
        log_path = None
    else:
        log_path = args["output_root"] / "logs" / f"{video_id}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(video_id, log_path)
    logger.info(f"Processing video {video_path}")
    logger.info(f"Steps: {steps}")

    meta = {}
    meta_path = args["output_root"] / "meta" / f"{video_id}.json"
    if meta_path.exists():
        meta = load_meta(meta_path)
        logger.info(f"Loaded meta data from {meta_path}")

    update_meta = False
    for step in all_steps:
        if step not in steps:
            force_step = False
        elif step in steps and steps[step]:
            force_step = True
        else:
            force_step = force

        video_id = video_path.stem
        output_dir = args["output_root"] / step / video_id
        func_meta_path = output_dir / "meta.json"

        if not force_step and step in meta:
            logger.info(f"Loaded {step} meta from {meta_path}")
            func_meta = meta[step]
        elif not force_step and func_meta_path.exists():
            logger.info(f"Loaded {step} meta from {func_meta_path}")
            func_meta = load_meta(func_meta_path)
            update_meta = True
        else:
            func_meta = {}

        if not func_meta and step in steps:
            if return_completion:
                return False
            if DEBUG:
                output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Running step {step}")
            func = globals()[step]
            func(
                video_path,
                args.get(step, {}),
                func_meta,
                meta,
                output_dir,
                logger,
            )
            if func_meta:
                update_meta = True
                save_meta(func_meta_path, func_meta)

        if func_meta:
            meta[step] = func_meta
            if not func_meta.get("pass", True):
                logger.info(f"Step {step} rejected video {video_path}")
                break

    if return_completion:
        return True

    if update_meta and meta:
        save_meta(meta_path, meta)
        logger.info(f"Saved meta data to {meta_path}")
        for step in all_steps:
            func_folder = args["output_root"] / step / video_id
            if func_folder.exists() and func_folder.is_dir():
                for file in func_folder.glob("*.json"):
                    file.unlink()
                    logger.info(f"Deleted {file}")
                if not any(func_folder.iterdir()):
                    func_folder.rmdir()
                    logger.info(f"Deleted {func_folder}")
    return meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--steps", type=str, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    cli_args = parser.parse_args()
    video_path = Path(cli_args.video_path)
    process_video(
        video_path,
        steps=cli_args.steps,
        force=cli_args.force
    )


if __name__ == "__main__":
    main()
