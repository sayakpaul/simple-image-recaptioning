import webdataset as wds
from functools import partial
from huggingface_hub.utils import insecure_hashlib
import torch
import os


def preprocess_fn(sample):
    image = sample["jpg"]
    image_hash = insecure_hashlib.sha1(image.tobytes()).hexdigest()
    sample_dict = {"original_image": image, "img_hash": image_hash}
    if "txt" in sample:
        sample_dict.update({"original_caption": sample["txt"]})
    else:
        sample_dict.update({"original_caption": "No caption present."})
    return sample_dict


# Taken from https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/01a4ab54307b9156c527d45b6b171f88623d2dec/imagenet.py#L65.
def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
    else:
        yield from src


class ExistsFilter:
    def __init__(self, output_dir):
        self.current_training_img_hashes = set()
        if os.path.isdir(output_dir):
            self.current_training_img_hashes = {
                f.split(".jpg")[0] for f in os.listdir(output_dir) if f.endswith(".jpg")
            }
        if self.current_training_img_hashes:
            print(f"Existing images found: {len(self.current_training_img_hashes)}.")

    def __call__(self, x):
        return x["img_hash"] not in self.current_training_img_hashes if self.current_training_img_hashes else True


def collate_fn(examples, predictor=None, return_probs=False):
    inputs = {
        "original_images": [sample["original_image"] for sample in examples],
        "original_captions": [example["original_caption"] for example in examples],
        "img_hashes": [example["img_hash"] for example in examples],
    }

    if predictor is not None:
        # `num_workers=0` because this is going to be already run under a dataloader. Cannot create workers
        # under workers.
        results = predictor.run(inputs["original_images"], num_workers=0, bs=16, pbar=False, return_probs=return_probs)
        # In this case, we filter out watermarked images.
        if return_probs and return_probs != "scores":
            inputs = {key: [item for item, result in zip(inputs[key], results) if result == "clean"] for key in inputs}
        # In this case, we simply return the scores.
        elif return_probs == "scores":
            inputs.update({"watermark_scores": results.tolist()})

    return inputs


def get_dataset(data_path, batch_size, output_dir, detect_watermarks=False):
    predictor = None
    if detect_watermarks:
        from wmdetection.models import get_watermarks_detection_model
        from wmdetection.pipelines.predictor import WatermarksPredictor
        from onnxruntime import SessionOptions, GraphOptimizationLevel

        # Otherwise, SLURM can cry.
        session_options = SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        transforms = get_watermarks_detection_model(
            "convnext-tiny", fp16=False, device="cpu", return_transforms_only=True
        )
        # We hard-code the ONNX model path (bad I know).
        predictor = WatermarksPredictor(
            "convnext.onnx", transforms, use_onnx=True, device="cpu", session_options=session_options
        )
        # when `detect_watermarks="scores"`,
        # we return the softmax scores associated to the "watermarked" classes.
        return_probs = detect_watermarks
    else:
        return_probs = False

    dataset = (
        wds.WebDataset(data_path, handler=wds.warn_and_continue, nodesplitter=nodesplitter, empty_check=False)
        .decode("pil", handler=wds.warn_and_continue)
        .map(preprocess_fn, handler=wds.warn_and_continue)
    )
    filter_obj = ExistsFilter(output_dir)
    if filter_obj.current_training_img_hashes:
        dataset = dataset.select(filter_obj)
    return dataset.batched(
        batch_size, partial=False, collation_fn=partial(collate_fn, predictor=predictor, return_probs=return_probs)
    )


def initialize_dataloader(data_path, batch_size, dataloader_num_workers, output_dir, detect_watermarks=False):
    print(f"Initializing dataloader with {data_path=}")
    dataset = get_dataset(
        data_path=data_path, batch_size=batch_size, output_dir=output_dir, detect_watermarks=detect_watermarks
    )
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        persistent_workers=True,
        num_workers=dataloader_num_workers,
        prefetch_factor=4,
    )
    print(f"Dataloader initialized with {dataloader_num_workers} workers")
    return dataloader
