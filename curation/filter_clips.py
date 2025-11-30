import json
from settings import args, get_logger, DEBUG
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
from process_video import video_montage, cut_video
import os


logger = get_logger(__name__, colored=False)

args |= {
    "fisheye_thres": 0.7,
    "3d_thres": 0.7,

    "motion_score_thres": 2.,
    "watermark_score_thres": 0.3,
    "camera_distance_thres": 0.005,
    "forward_degree_thres": 10,
    "up_degree_thres": 2,
    "max_num_clips_per_video": 5,

    "motion_score_bin_width": 1,
    "watermark_score_bin_width": 0.1,
    "slam_pose_bin_width": 0.001,

    "split_ratio": 0.9,
}


def main():
    meta_root = args["data_root"] / "meta"
    meta_files = list(meta_root.glob("*.json"))
    meta_files.sort()
    logger.info(f"Found {len(meta_files)} meta files")

    summary = {
        "total_clips": 0,
        "slam_clip_num": defaultdict(lambda: 0),
        "durations": defaultdict(lambda: []),
        "outliers": defaultdict(lambda: []),
        "motion_scores": defaultdict(lambda: []),
        "watermark_scores": defaultdict(lambda: []),
        "camera_distances": defaultdict(lambda: []),
        "motion_score_bins": defaultdict(lambda: 0),
        "watermark_score_bins": defaultdict(lambda: 0),
        "slam_pose_bins": defaultdict(lambda: 0),
        "up_degree_diffs": defaultdict(lambda: []),
        "forward_degree_diffs": defaultdict(lambda: []),
        "fisheye_scores": defaultdict(lambda: []),
        "3d_scores": defaultdict(lambda: []),
    }
    if DEBUG:
        clip_folder = args["debug_root"] / "filter_clips"
        clip_folder.mkdir(parents=True, exist_ok=True)

    def save_sample(clip, sample_path):
        if sample_path.exists():
            return

        if "video_name" in clip:
            src_path = args["data_root"] / "export_clips" / clip["video_id"] / clip["video_name"]
        elif "clip_id" not in clip:
            src_path = (args["data_root"] / "videos" / clip["video_id"]).with_suffix(".mp4")
        else:
            src_path = None

        if src_path and src_path.exists():
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            os.link(src_path, sample_path)
            return

        video_path = args["data_root"] / "videos" / clip["video_id"]
        video_path = video_path.with_suffix(".mp4")
        cut_video(video_path, clip["frames"][0], clip["frames"][1], sample_path)
        # montage_path = sample_path.with_suffix(".gif")
        # video_montage(sample_path, montage_path, num_frames=48, fps=24)

    filtered_clips = {
        "Fisheye format": [],
        "3D format": [],
        "Success": [],
        "No camera pose": [],
        "Small camera motion": [],
        "Small camera translation": [],
        "Small camera rotation": [],
        "Small motion score": [],
        "Large watermark score": [],
    }
    for meta_file in tqdm(meta_files):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        if "format_check" in meta:
            video_dict = {
                "video_id": meta_file.stem,
            }
            format_scores = meta["format_check"]["scores"]
            for key in format_scores:
                summary_key = f"{key}_scores"
                if summary_key in summary:
                    summary[summary_key]["Success"].append(format_scores[key])
            if format_scores["fisheye"] > args["fisheye_thres"]:
                filtered_clips["Fisheye format"].append(video_dict)
                continue
            if format_scores["3d"] > args["3d_thres"]:
                filtered_clips["3D format"].append(video_dict)
                continue

        if "slam_clips" not in meta:
            continue
        clips = meta["slam_clips"]["clips"]
        summary["total_clips"] += len(clips)

        outlier_sample = True
        score_clip_sample = True
        clips = np.random.permutation(clips)
        num_clips = 0
        for clip in clips:
            clip_dict = {
                "video_id": meta_file.stem,
                "clip_id": clip["clip_id"],
                "frames": clip["frames"],
                "clip_name": clip["clip_name"],
            }
            if "export_clips" in meta:
                clip_dict["video_name"] = meta["export_clips"]["clips"][clip["clip_id"] - 1]["video_name"]

            summary["slam_clip_num"][clip["info"]] += 1
            duration = clip["seconds"][1] - clip["seconds"][0]
            summary["durations"][clip["info"]].append(duration)

            if outlier_sample:
                if duration < args["slam_clips"]["min_clip_len"] or duration > args["slam_clips"]["max_clip_len"]:
                    summary["outliers"][clip["info"]].append(clip | {
                        "video_id": meta_file.stem,
                        "duration": duration,
                    })
                outlier_sample = False

            if "watermark_score" in meta:
                watermark_score = meta["watermark_score"]["clips"][clip["clip_id"] - 1]["score"]
                summary["watermark_scores"][clip["info"]].append(watermark_score)
                if watermark_score > args["watermark_score_thres"]:
                    filtered_clips["Large watermark score"].append(clip_dict)
                    continue

            if "motion_score" in meta:
                motion_score = meta["motion_score"]["clips"][clip["clip_id"] - 1]["score"]
                summary["motion_scores"][clip["info"]].append(motion_score)
                if motion_score < args["motion_score_thres"]:
                    filtered_clips["Small motion score"].append(clip_dict)
                    continue

            if "slam_pose" in meta:
                pose_clip = meta["slam_pose"]["clips"][clip["clip_id"] - 1]
                pose_info = pose_clip["info"]
                if pose_info == "Success":
                    clip_name = clip["clip_name"]
                    camera_poses = (args["data_root"] / "slam_pose" / meta_file.stem / clip_name).with_suffix(".npy")
                    extrinsics = np.load(camera_poses)
                    camera_pos = extrinsics[:, :3, 3]
                    camera_distance = np.linalg.norm(camera_pos[1:] - camera_pos[:-1], axis=1)
                    camera_distance = float(np.mean(camera_distance))
                    summary["camera_distances"][clip["info"]].append(camera_distance)
                    pose_clip["score"] = camera_distance
                    if camera_distance < args["camera_distance_thres"]:
                        filtered_clips["Small camera translation"].append(clip_dict)
                        continue

                    ref_up = - extrinsics[0, :3, 1]
                    ref_forward = extrinsics[0, :3, 2]
                    src_up = - extrinsics[1:, :3, 1]
                    src_forward = extrinsics[1:, :3, 2]
                    up_cos_angle = np.clip(np.dot(src_up, ref_up), -1, 1)
                    up_degree_diff = np.arccos(up_cos_angle) * 180 / np.pi
                    forward_cos_angle = np.clip(np.dot(src_forward, ref_forward), -1, 1)
                    forward_degree_diff = np.arccos(forward_cos_angle) * 180 / np.pi

                    sample_idx = np.random.choice(
                        extrinsics.shape[0] - 1,
                        size=3,
                        replace=False,
                    )
                    summary["up_degree_diffs"]["Success"].extend(up_degree_diff[sample_idx].tolist())
                    summary["forward_degree_diffs"]["Success"].extend(forward_degree_diff[sample_idx].tolist())

                    if up_degree_diff.mean() < args["up_degree_thres"] or forward_degree_diff.mean() < args["forward_degree_thres"]:
                        filtered_clips["Small camera rotation"].append(clip_dict)
                        continue
                    
                elif clip["info"] == "Success":
                    filtered_clips["No camera pose"].append(clip_dict)
                    continue

            filtered_clips[clip["info"]].append(clip_dict)

            if DEBUG and score_clip_sample:
                score_clip_sample = False
                for score_key in ["motion_score", "watermark_score", "slam_pose"]:
                    if score_key not in meta:
                        continue
                    score = meta[score_key]["clips"][clip["clip_id"] - 1].get("score", None)
                    if score is None:
                        continue
                    score_bin_left = int(score / args[f"{score_key}_bin_width"]) * args[f"{score_key}_bin_width"]
                    score_bin_right = score_bin_left + args[f"{score_key}_bin_width"]
                    score_bin = (f"{score_bin_left:.4f}", f"{score_bin_right:.4f}")
                    if summary[f"{score_key}_bins"][score_bin] < 1:
                        summary[f"{score_key}_bins"][score_bin] += 1
                        sample_path = \
                            clip_folder / \
                            f"score_samples - {score_key.replace('_', ' ').title()}" / \
                            f"({score_bin[0]} - {score_bin[1]}) {score:.4f} - {meta_file.stem}-Clip-{clip["clip_id"]:03d}.mp4"
                        save_sample(clip_dict, sample_path)

            if clip["info"] == "Success":
                num_clips += 1
                if num_clips >= args["max_num_clips_per_video"]:
                    break

    logger.info("[Summary]")
    logger.info(f"Total videos: {len(meta_files)}")
    logger.info(f"Total clips: {summary['total_clips']}")

    logger.info("[SLAM Clips]")
    score_keys = [
        "durations", "motion_scores", "watermark_scores", "camera_distances",
        "up_degree_diffs", "forward_degree_diffs", "fisheye_scores", "3d_scores",
    ]
    for info, count in summary["slam_clip_num"].items():
        logger.info(f"{info}: {count}")

        for score_key in score_keys:
            if not summary[score_key] or info not in summary[score_key]:
                continue
            scores = summary[score_key][info]
            score_name = score_key.replace("_", " ").title()
            logger.info(f"{info} - avg. {score_name}: {sum(scores) / len(scores):.4f}")
            logger.info(f"{info} - min. {score_name}: {min(scores):.4f}")
            logger.info(f"{info} - max. {score_name}: {max(scores):.4f}")
            logger.info(f"{info} - 10th percentile {score_name}: {np.percentile(scores, 10):.4f}")
            logger.info(f"{info} - 90th percentile {score_name}: {np.percentile(scores, 90):.4f}")

    if summary["outliers"]:
        logger.info("[Duration Outliers]")
        for info, clips in summary["outliers"].items():
            logger.info(f"{info}: {len(clips)}")
            for clip in clips[:10]:
                logger.info(f"{info} - {clip["video_id"]} - {clip["clip_id"]}: {clip["duration"]:.2f} seconds")

    logger.info("[Filtered Clips]")
    output_folder = args["output_root"] / "filter_clips"
    output_folder.mkdir(parents=True, exist_ok=True)
    for info, clips in filtered_clips.items():
        list_name = info.replace(" ", "_").lower()
        list_path = output_folder / f"{list_name}.json"
        with open(list_path, "w") as f:
            json.dump(clips, f, indent=4)
        logger.info(f"{info}: {len(clips)}, saved to {list_path}")

        if DEBUG:
            logger.debug(f"Exporting samples for {info}")
            num_samples = 100 if info == "Success" else 3
            clips_sample = np.random.choice(clips, size=min(num_samples, len(clips)), replace=False)
            for clip in clips_sample:
                sample_name = f"{clip["video_id"]}"
                if "clip_id" in clip:
                    sample_name += f"-Clip-{clip["clip_id"]:03d}"
                sample_path = (clip_folder / f"filter_samples - {info}" / sample_name).with_suffix(".mp4")
                save_sample(clip, sample_path)

    logger.info("[Split Data]")
    video_ids = [meta_file.stem for meta_file in meta_files]
    split_idx = int(len(video_ids) * args["split_ratio"])
    train_video_ids = set(video_ids[:split_idx])
    train_clips = []
    test_clips = []
    for clips in filtered_clips["Success"]:
        if clips["video_id"] in train_video_ids:
            train_clips.append(clips)
        else:
            test_clips.append(clips)
    train_list_path = output_folder / "train.json"
    test_list_path = output_folder / "test.json"
    with open(train_list_path, "w") as f:
        json.dump(train_clips, f, indent=4)
    with open(test_list_path, "w") as f:
        json.dump(test_clips, f, indent=4)
    logger.info(f"Train clips: {len(train_clips)}, {len(train_clips) / (len(train_clips) + len(test_clips)):.2%} of total clips, saved to {train_list_path}")
    logger.info(f"Test clips: {len(test_clips)}, {len(test_clips) / (len(train_clips) + len(test_clips)):.2%} of total clips, saved to {test_list_path}")

    if DEBUG:
        import matplotlib.pyplot as plt
        logger.debug("[Histograms]")
        histogram_folder = clip_folder / "histograms"
        histogram_folder.mkdir(parents=True, exist_ok=True)
        for score_key in score_keys:
            score_name = score_key.replace("_", " ").title()
            samples = summary[score_key]
            if not samples:
                continue
            samples = np.concatenate(list(samples.values()))
            samples = samples[(samples > np.percentile(samples, 5)) & (samples < np.percentile(samples, 95))]
            plt.hist(samples, bins=30)
            plt.ylabel("Count")
            plt.xlabel(score_name)
            plt.title(f"{score_name} Histogram")
            histogram_path = histogram_folder / f"{score_key}_histogram.png"
            plt.savefig(histogram_path)
            plt.close()
            logger.debug(f"Saved {score_key} histogram to {histogram_path}")


if __name__ == '__main__':
    main()
