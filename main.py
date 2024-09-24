import os
import queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Union
import fire

from data_processing import initialize_dataloader
from model import load_vllm_engine, infer
from utils import save_results


def main(
    data_path: str,
    batch_size: int = 48,
    dataloader_num_workers: int = 8,
    output_dir: str = "sample_outputs",
    max_tokens: int = 120,
    detect_watermarks: Union[bool, str] = False,
):
    vllm_engine, sampling_params = load_vllm_engine(max_tokens=max_tokens)

    dataloader = initialize_dataloader(
        data_path=data_path,
        batch_size=batch_size,
        dataloader_num_workers=dataloader_num_workers,
        output_dir=output_dir,
        detect_watermarks=detect_watermarks,
    )

    output_queue = queue.Queue()
    save_thread = ThreadPoolExecutor(max_workers=dataloader_num_workers)
    os.makedirs(output_dir, exist_ok=True)
    save_future = save_thread.submit(save_results, output_queue, output_dir)

    try:
        print("Starting the generation process.")
        for batch in tqdm(dataloader):
            batch["sampling_params"] = sampling_params
            outputs = infer(vllm_engine, batch)

            original_captions = batch["original_captions"]
            original_images = batch["original_images"]
            image_hashes = batch["img_hashes"]

            if detect_watermarks:
                if detect_watermarks != "scores":
                    output_queue.put((original_captions, outputs, original_images, image_hashes))
                else:
                    watermark_scores = batch["watermark_scores"]
                    output_queue.put((original_captions, outputs, original_images, image_hashes, watermark_scores))
            else:
                output_queue.put((original_captions, outputs, original_images, image_hashes))

    finally:
        output_queue.put(None)
        save_thread.shutdown(wait=True)

    save_future.result()
    print("All processes completed. Captions generation and saving done.")


if __name__ == "__main__":
    fire.Fire(main)
