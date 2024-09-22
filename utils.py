# Pillow decoding errors for large files.
# Reference: https://stackoverflow.com/questions/42671252/python-pillow-valueerror-decompressed-data-too-large
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import json
import os
from PIL import Image
import io
import queue


def save_results(output_queue, output_dir):
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break

            if len(item) == 4:
                original_captions, outputs, img_bytes, img_hashes = item
            else:
                original_captions, outputs, img_bytes, img_hashes, watermark_scores = item
            
            outputs = [o.outputs[0].text for o in outputs]

            for i, caption in enumerate(original_captions):
                original_image = Image.open(io.BytesIO(img_bytes[i])).convert("RGB")
                img_path = os.path.join(output_dir, f"{img_hashes[i]}.jpg")
                original_image.save(img_path)

                caption_dict = {"original": caption, "predicted": outputs[i]}
                with open(os.path.join(output_dir, f"{img_hashes[i]}_caption.json"), "w") as f:
                    json.dump(caption_dict, f, indent=4)
                
                if watermark_scores is not None:
                    watermark_score = watermark_scores[i]
                    watermark_score_dict = {"watermark_score": str(watermark_score)}
                    with open(os.path.join(output_dir, f"{img_hashes[i]}_watermark.json"), "w") as f:
                        json.dump(watermark_score_dict, f, indent=4)

        except queue.Empty:
            continue
