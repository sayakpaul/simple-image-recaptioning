import json
import os
from PIL import Image
import io


def save_results(output_queue, output_dir):
    while True:
        try:
            item = output_queue.get(timeout=5)
            if item is None:
                break

            original_captions, outputs, img_bytes, img_hashes = item
            outputs = [o.outputs[0].text for o in outputs]

            for caption, pred_caption, img_byte, img_hash in zip(original_captions, outputs, img_bytes, img_hashes):
                original_image = Image.open(io.BytesIO(img_byte)).convert("RGB")
                img_path = os.path.join(output_dir, f"{img_hash}.jpg")
                original_image.save(img_path)

                caption_dict = {"original": caption, "predicted": pred_caption}
                with open(os.path.join(output_dir, f"{img_hash}_caption.json"), "w") as f:
                    json.dump(caption_dict, f, indent=4)

        except queue.Empty:
            continue
