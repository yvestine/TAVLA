from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
from lerobot.common.datasets.compute_stats import aggregate_stats
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.shared.effort_type import EffortType
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        return (
            len(self._dataset.hf_dataset) if self._dataset.hf_dataset is not None else self._dataset.meta.total_frames
        )


class WeightedMixtureDataset(Dataset):
    """Sample multiple datasets with a fixed per-epoch source ratio.

    LeRobot's MultiLeRobotDataset concatenates datasets, which makes sampling
    proportional to raw frame counts. TAVLA fine-tuning needs an explicit
    90/10 real/simulation ratio, so this wrapper builds a deterministic source
    schedule and samples a frame from the selected source with replacement.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        seed: int = 0,
        disabled_features: Sequence[str] = (),
    ):
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same non-zero length")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("Cannot mix an empty dataset")

        weights_array = np.asarray(weights, dtype=np.float64)
        if np.any(weights_array <= 0) or not np.isfinite(weights_array).all():
            raise ValueError(f"dataset_sampling_weights must be finite and positive: {weights}")
        weights_array /= weights_array.sum()

        # Keep the largest common number of samples that does not require
        # discarding the smaller source, then draw source frames with replacement.
        total_samples = int(min(len(dataset) / weight for dataset, weight in zip(datasets, weights_array, strict=True)))
        if total_samples < len(datasets):
            raise ValueError(f"Datasets are too small for weights={weights}: {[len(d) for d in datasets]}")
        counts = np.floor(total_samples * weights_array).astype(np.int64)
        counts[-1] += total_samples - int(counts.sum())

        rng = np.random.default_rng(seed)
        source_ids = np.concatenate([np.full(count, index, dtype=np.int64) for index, count in enumerate(counts)])
        rng.shuffle(source_ids)
        local_indices = np.asarray(
            [rng.integers(len(datasets[source_id])) for source_id in source_ids],
            dtype=np.int64,
        )

        self._datasets = list(datasets)
        self._source_ids = source_ids
        self._local_indices = local_indices
        self.source_counts = counts
        self.weights = weights_array
        self.disabled_features = frozenset(disabled_features)

    def __len__(self) -> int:
        return len(self._source_ids)

    def __getitem__(self, index: SupportsIndex):
        index = index.__index__()
        source_id = int(self._source_ids[index])
        item = self._datasets[source_id][int(self._local_indices[index])]
        for key in self.disabled_features:
            item.pop(key, None)
        return item


class NoEffortMultiLeRobotDataset(lerobot_dataset.MultiLeRobotDataset):
    """Multi-dataset wrapper that removes stored effort from no-effort training.

    LeRobot's ``MultiLeRobotDataset`` aggregates every stored feature's metadata
    before it applies the requested timestamp queries.  This makes a real dataset
    with 7-D joint effort and a simulation dataset with 6-D wrench effort fail to
    initialize, even when the Pi0 configuration does not use effort at all.

    The effort feature is disabled only in memory and is removed from each sample
    at the wrapper boundary. Dataset files and their metadata on disk are unchanged.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerances_s=None,
        *,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        self.repo_ids = repo_ids
        self.root = Path(root) if root is not None else Path(lerobot_dataset.LEROBOT_HOME)
        self.tolerances_s = tolerances_s if tolerances_s else {repo_id: 1e-4 for repo_id in repo_ids}
        self._datasets = [
            lerobot_dataset.LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                local_files_only=local_files_only,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Keep only keys that can be collated across datasets, and always disable
        # effort because this wrapper is selected only for EffortType.NO configs.
        self.disabled_features = {"observation.effort"}
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError("Multiple datasets were provided but they had no keys common to all of them.")
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    "keys %s of %s were disabled as they are not contained in all the other datasets.",
                    extra_keys,
                    repo_id,
                )
            self.disabled_features.update(extra_keys)

        # Prevent LeRobot's aggregate_stats from touching incompatible effort
        # vectors. This mutation is local to the in-memory metadata object.
        for ds in self._datasets:
            ds.meta.stats.pop("observation.effort", None)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats(self._datasets)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    repo_ids = list(repo_id) if isinstance(repo_id, list | tuple) else None
    if repo_ids is None:
        dataset_class = lerobot_dataset.LeRobotDataset
    else:
        dataset_class = (
            NoEffortMultiLeRobotDataset
            if getattr(model_config, "effort_type", EffortType.NO) == EffortType.NO
            else lerobot_dataset.MultiLeRobotDataset
        )
    # NOTE here we assume all repos have the same fps.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_ids[0] if repo_ids is not None else repo_id,
        local_files_only=data_config.local_files_only,
    )

    delta_timestamps = {
        **{
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        }
    }
    # Do not register an empty effort query in no-effort mode. LeRobot tries to
    # convert every queried field to torch; querying an empty timestamp list
    # produces an empty list and crashes in hf_transform_to_torch.
    if data_config.effort_history:
        delta_timestamps["observation.effort"] = [t / dataset_meta.fps for t in data_config.effort_history]

        if getattr(model_config, "effort_type", EffortType.NO) in (
            EffortType.EXPERT_FUT,
            EffortType.EXPERT_HIS_C_FUT,
            EffortType.EXPERT_HIS_C_L_FUT,
        ):
            delta_timestamps["observation.effort"] += [
                (t + 1) / dataset_meta.fps for t in range(model_config.action_horizon)
            ]

    dataset = dataset_class(
        repo_id,
        delta_timestamps=delta_timestamps,
        local_files_only=data_config.local_files_only,
    )

    dataset_sources = None
    if repo_ids is not None:
        dataset_sources = getattr(dataset, "_datasets", None)
        if dataset_sources is None:
            raise TypeError("Multi-dataset loader must expose its source datasets")

    if data_config.prompt_from_task:
        if repo_ids is None:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        else:
            for idx, repo in enumerate(repo_ids):
                dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
                    repo, local_files_only=data_config.local_files_only
                )
                dataset_sources[idx] = TransformedDataset(
                    dataset_sources[idx], [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
                )

    if repo_ids is not None and data_config.dataset_sampling_weights:
        dataset = WeightedMixtureDataset(
            dataset_sources,
            data_config.dataset_sampling_weights,
            seed=42,
            disabled_features=getattr(dataset, "disabled_features", ()),
        )

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        print(f"data_config:{data_config}")
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
