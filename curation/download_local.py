import os
import subprocess
import pandas as pd
import argparse

def download_video(
    video_id: str,
    output_file: str,
    max_height: int = 1000,
    include_audio: bool = False,
    start_time: int = None,
    end_time: int = None,
    fps: int = None,
    quiet: bool = True,
    surpress_output: bool = True,
):
    if os.path.exists(output_file):
        print(f"Video {video_id} already exists at {output_file}. Skipping download.")
        return

    dirname = os.path.dirname(output_file)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)


    format_str = "bestvideo[ext=mp4][height=720]"

    command = [
        "conda",
        "run",
        "-n",
        "panflow_curation",
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
    subprocess.run(
        command,
        stdout=output_destination,
        stderr=output_destination,
    )

    print(f"Downloaded {video_id} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos from a parquet file.")
    parser.add_argument('--in_path', required=True)
    parser.add_argument('--out_dir', required=True)
    
    args = parser.parse_args()

    parquet_path = args.in_path
    output_folder = args.out_dir
    
    df = pd.read_parquet(parquet_path)
    yt_ids = df["id"].unique().tolist()
    os.makedirs(output_folder, exist_ok=True)

    for idx, video_id in enumerate(yt_ids, start=1):
        output_file = os.path.join(output_folder, f"{video_id}.mp4")
        print(f"Downloading {idx}/{len(yt_ids)}: {video_id}")
        download_video(video_id, output_file=output_file)
