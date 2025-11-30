from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional, Literal
import csv
import json
import decord
from tqdm.auto import tqdm

from .base_dataset import BaseDataset, TestDataArgs


class Web360Dataset(BaseDataset):
    def load_split(self):
        split_path = self.args.data_root / "WEB360_360TF_train.json"
        if not split_path.exists():
            csv_path = split_path.with_suffix(".csv")
            with open(csv_path, "r") as csvfile:
                csv_dataset = list(csv.DictReader(csvfile))
            dataset = []
            for data in tqdm(csv_dataset, desc="Converting CSV to JSON"):
                video_path = str((self.args.data_root / "videos_512x1024x100" / data["videoid"]).with_suffix(".mp4"))
                video = decord.VideoReader(video_path, num_threads=1)
                dataset.append({
                    "video_id": data["videoid"],
                    "clip_id": 1,
                    "clip_name": "Clip-001",
                    "frames": (0, len(video) - 1),
                    "caption": data["name"],
                })
            with open(split_path, "w") as f:
                json.dump(dataset, f, indent=4)

        with open(split_path, "r") as f:
            dataset = json.load(f)
        for data in dataset:
            data["video_path"] = str((self.args.data_root / "videos_512x1024x100" / data["video_id"]).with_suffix(".mp4"))
        self.dataset = dataset


if __name__ == "__main__":
    from .base_dataset import test_dataset

    test_dataset(
        cls = Web360Dataset,
        args = TestDataArgs(
            data_root = Path("data/WEB360M"),
            derotation = "no",
        ),
        keys = [
            "video", "image", "flow", "noise",
        ],
    )
