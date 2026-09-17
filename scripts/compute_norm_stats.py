"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from etils import epath
import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class InjectDummyImages(transforms.DataTransformFn):
    """Provide tiny placeholders after image columns are removed for stats-only loading."""

    def __init__(self, image_keys: tuple[str, ...]):
        self._image_keys = image_keys

    def __call__(self, x: dict) -> dict:
        for key in self._image_keys:
            x[key] = np.zeros((3, 1, 1), dtype=np.uint8)
        return x


def optimize_stats_dataset(dataset: _data_loader.Dataset) -> list[transforms.DataTransformFn]:
    """Skip decoding embedded images, which are not used for normalization stats."""
    hf_dataset = getattr(dataset, "hf_dataset", None)
    meta = getattr(dataset, "meta", None)
    if hf_dataset is None or meta is None:
        return []
    if meta.video_keys:
        return []

    image_keys = tuple(meta.camera_keys)
    if not image_keys:
        return []

    dataset.hf_dataset = hf_dataset.remove_columns(image_keys)
    return [InjectDummyImages(image_keys)]


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    stats_only_transforms = optimize_stats_dataset(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *stats_only_transforms,
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None, batch_size: int = 8, num_workers: int = 0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if isinstance(data_config.repo_id, str):
        datasets = [data_config.repo_id]
    elif isinstance(data_config.repo_id, list):
        print("compute norm stats for multiple repos: ", data_config.repo_id)
        datasets = data_config.repo_id
    else:
        raise ValueError("Data config must have a repo_id")

    for data_id in datasets:
        print(f"compute norm stats for repo: {data_id}")
        object.__setattr__(data_config, "repo_id", data_id)

        dataset = _data_loader.create_dataset(data_config, config.model)
        stats_only_transforms = optimize_stats_dataset(dataset)
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *stats_only_transforms,
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
                RemoveStrings(),
            ],
        )

        num_frames = len(dataset)
        shuffle = False

        if max_frames is not None and max_frames < num_frames:
            num_frames = max_frames
            shuffle = True

        # TorchDataLoader drops incomplete batches, so use complete batches only
        # and avoid looping back over the dataset to fill a partial batch.
        num_batches = num_frames // batch_size
        if num_batches < 1:
            raise ValueError("The dataset contains no frames to compute statistics from.")

        data_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            num_batches=num_batches,
        )

        keys = ["state", "actions"]
        if getattr(config.data, "effort_history", False):
            keys.append("effort")
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                values = np.asarray(batch[key])
                stats[key].update(values.reshape(-1, values.shape[-1]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        # Respect an explicitly configured assets directory.  Multi-dataset
        # TAVLA configs share one assets root per sampling ratio, so writing
        # only to config.assets_dirs would make the trainer unable to find the
        # freshly computed stats.
        assets_root = epath.Path(config.data.assets.assets_dir or config.assets_dirs)
        output_path = assets_root / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
