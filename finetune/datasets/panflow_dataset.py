from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional, Literal
import json
from .base_dataset import BaseDataset, TestDataArgs


class PanFlowDataset(BaseDataset):
    def load_split(self):
        split_path = (self.args.data_root / "filter_clips" / self.split).with_suffix(".json")
        with open(split_path, "r") as f:
            dataset = json.load(f)
        for data in dataset:
            data["video_path"] = str((self.args.data_root / "videos" / data["video_id"]).with_suffix(".mp4"))
        self.dataset = dataset


if __name__ == "__main__":
    from .base_dataset import test_dataset

    # test_dataset(
    #     cls = PanFlowDataset,
    #     args = TestDataArgs(
    #         data_root = Path("data/PanFlow/"),
    #         derotation = "no",
    #         demo_name = "editing",
    #     ),
    #     keys = [
    #         "video", "image", "flow", "noise",
    #     ],
    # )

    # test_dataset(
    #     cls = PanFlowDataset,
    #     args = TestDataArgs(
    #         data_root = Path("data/PanFlow/"),
    #         derotation = "no",
    #         demo_name = "motion_transfer",
    #     ),
    #     keys = [
    #         "video", "image", "flow", "noise",
    #     ],
    # )

    test_dataset(
        cls = PanFlowDataset,
        args = TestDataArgs(
            data_root = Path("data/PanFlow/"),
        ),
    )
