import os

from datasets import interleave_datasets, load_dataset, load_from_disk


def exist_and_not_none(d, key):
    return key in d and d[key] is not None


def blending_datasets(
    datasets,
    probabilities=None,
    strategy=None,
    seed=42,
    max_count=100000000,
    stopping_strategy="all_exhausted",
    dataset_split="train",
    is_rank_0=None,
):
    if strategy is not None and not hasattr(strategy, "is_rank_0"):
        seed = strategy
        strategy = None
    if is_rank_0 is None:
        is_rank_0 = True if strategy is None else strategy.is_rank_0()

    datasets = datasets.split(",")
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets)

    data_list = []
    for dataset in datasets:
        dataset = dataset.strip()
        if is_rank_0:
            print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        data_name = dataset.split("#")[1].strip() if "#" in dataset else None
        dataset = dataset.split("#")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        if ext == ".py" or (os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))):
            data = load_dataset(dataset, trust_remote_code=True)
            if is_rank_0:
                print(f"loaded {dataset} with python script")
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            if is_rank_0:
                print(f"loaded {dataset} with data_files={dataset}")
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                if is_rank_0:
                    print(f"loaded {dataset} from disk")
            except Exception as e:
                if is_rank_0:
                    print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                if is_rank_0:
                    print(f"loaded {dataset} from files")
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            if is_rank_0:
                print(f"loaded {dataset} from files")

        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    if is_rank_0:
        print(data_list)

    if probabilities is None:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(data_list)
    else:
        dataset = interleave_datasets(
            data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    return dataset
