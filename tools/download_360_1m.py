import os
import subprocess
import argparse
import json
from tqdm.auto import tqdm


def download_video(
    video_id: str,
    output_file: str,
    max_height: int = 1000,
    include_audio: bool = False,
    start_time: int = None,
    end_time: int = None,
    fps: int = None,
    quiet: bool = False,
    surpress_output: bool = False,
    retry: int = 3,
):
    if os.path.exists(output_file):
        tqdm.write(f"Video {video_id} already exists at {output_file}. Skipping download.")
        return

    dirname = os.path.dirname(output_file)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    format_str = "bestvideo[ext=mp4][height=720]"

    command = [
        "conda",
        "run",
        "-n",
        "panflow",
        "yt-dlp",
        f"https://youtube.com/watch?v={video_id}",
        "-f", format_str,
        "-o", output_file,
        "--user-agent", ""
    ]

    if start_time is not None and end_time is not None:
        command.append("--download-sections")
        command.append(f"*{start_time}-{end_time}")

    if fps is not None:
        command.append("--downloader-args")
        command.append(f"ffmpeg:-filter:v fps={fps} -vcodec h264 -f mp4")


    if quiet:
        command.append("-q")


    output_destination = subprocess.DEVNULL if surpress_output else None
    for attempt in range(retry):
        try:
            subprocess.run(
                command,
                stdout=output_destination,
                stderr=output_destination,
                check=True,
            )
            break  # Exit loop if download succeeds
        except subprocess.CalledProcessError as e:
            if attempt < retry - 1:
                tqdm.write(f"Attempt {attempt + 1} failed for {video_id}. Retrying...")
            else:
                tqdm.write(f"Failed to download {video_id} after {retry} attempts.")
                return

    tqdm.write(f"Downloaded {video_id} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from a parquet file.")
    parser.add_argument('--in_path', default="data/360-1M/filter_clips/", type=str)
    parser.add_argument('--splits', default=["train", "test"], nargs='+', type=str)
    parser.add_argument('--out_dir', default="data/360-1M/videos/", type=str)
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        split_path = os.path.join(args.in_path, f"{split}.json")
        with open(split_path, "r") as f:
            data = json.load(f)
        video_ids = list(set(clip["video_id"] for clip in data))
        video_ids.sort()
        for idx, video_id in tqdm(enumerate(video_ids, start=1), total=len(video_ids)):
            output_file = os.path.join(args.out_dir, f"{video_id}.mp4")
            tqdm.write(f"Downloading {idx}/{len(video_ids)}: {video_id}")
            download_video(video_id, output_file=output_file)
