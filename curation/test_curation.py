import numpy as np
from settings import args, get_logger
from process_video import process_video


test_samples = [
    {
        "video_id": "--13wu6aHZU",
        "reject": False,
        "reason": "Fast moving camera",
    },
    {
        "video_id": "Rpw7eUs49sM",
        "reject": False,
        "reason": "Cholesky failure",
    },
    {
        "video_id": "mpx5CmPsP-k",
        "reject": False,
        "reason": "No features",
    },
    {
        "video_id": "IGBQpLlhB5Q",
        "reject": False,
        "reason": "No features",
    },
    {
        "video_id": "ofgBCmeu3-I",
        "reject": False,
        "reason": "No features",
    },
    {
        "video_id": "FVjp3ONwuPA",
        "reject": False,
        "reason": "Small camera motion",
    },
    {
        "video_id": "gJDrarJNcgA",
        "reject": False,
        "reason": "Small camera motion",
    },
    {
        "video_id": "HpV4rAgao-w",
        "reject": False,
        "reason": "Cholesky failure",
    },
    {
        "video_id": "mUVZ7rYDvUI",
        "reject": True,
        "reason": "Incomplete video file",
    },
    {
        "video_id": "KxueHpH80uc",
        "reject": True,
        "reason": "Invalid video file",
    },
    {
        "video_id": "__Oq9M9fj8Q",
        "reject": False,
        "reason": "Continuous",
    },
    {
        "video_id": "pVgPSj-7p6I",
        "reject": False,
        "reason": "Crossfade",
    },
    {
        "video_id": "__lCjsZGvYo",
        "reject": False,
        "reason": "Fade",
    },
    {
        "video_id": "bhwfBlxrwAc",
        "reject": True,
        "reason": "3D format",
    },
    {
        "video_id": "-BCRJVWa95I",
        "reject": True,
        "reason": "Fisheye format",
    }
]


def main():
    logger = get_logger(__name__)
    test_results = []
    for sample in test_samples:
        video_path = args["data_root"] / "videos" / f"{sample['video_id']}.mp4"
        if not video_path.exists():
            logger.info(f"Video file not found: {video_path}")
            continue
        logger.info(f"Processing video: {video_path}")
        meta = process_video(video_path)
        reject = not meta["format_check"]["pass"]
        if reject:
            logger.info(f"{meta["format_check"]["reason"]}, expected: {sample['reason']}")
        else:
            logger.info(f"Video is valid, expected: {'valid' if not sample['reject'] else 'invalid'}")

        passed = (reject == sample["reject"])
        test_results.append(passed)

    test_results = np.array(test_results)
    logger.info(f"Test results: {test_results.sum()}/{len(test_results)} passed")


if __name__ == "__main__":
    main()
