import json
import random

with open("/Users/itazaporozhets/VSCode/whatsup_vlms/data/controlled_images_dataset.json") as f:
    data = json.load(f)

all_captions_by_prep = {}
for item in data:
    first_caption = item["caption_options"][0]
    words = first_caption.split()
    preposition_index = next(i for i, w in enumerate(words) if w in {"on", "under", "right", "left"})
    preposition = words[preposition_index]
    all_captions_by_prep.setdefault(preposition, []).append(first_caption)

for item in data:
    first_caption = item["caption_options"][0]
    words = first_caption.split()
    preposition_index = next(i for i, w in enumerate(words) if w in {"on", "under", "right", "left"})
    preposition = words[preposition_index]
    filtered_captions = [c for c in all_captions_by_prep[preposition] if c != first_caption]
    item["caption_options"].extend(random.sample(filtered_captions, min(3, len(filtered_captions))))

with open("/Users/itazaporozhets/VSCode/whatsup_vlms/data/controlled_images_dataset_augmented.json", "w") as f:
    json.dump(data, f, indent=4)
