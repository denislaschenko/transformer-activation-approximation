from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    data_files={
        "validation": "data/validation-*.parquet"
    },
    split="validation",
    streaming=True
)

def collate_fn(samples):
    images = [sample["image"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    return images, labels


dataloader = DataLoader(
    dataset, # type: ignore
    batch_size=32,
    collate_fn=collate_fn
)


dataset = dataset.take(n=128)
def show_images(images, labels):
    plt.figure(figsize=(12, 12))

    for i in range(min(len(images), 16)):
        plt.subplot(4, 4, i + 1)

        plt.imshow(images[i])
        plt.title(dataset.features["label"].int2str(labels[i])) # type: ignore
        plt.axis("off")

    plt.tight_layout()
    plt.show()


images, labels = next(iter(dataloader))

show_images(images, labels)

# inputs = {}
# for image in dataset:
#     inputs.update(feature_extractor(images=image, return_tensors="pt"))

# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# assert model.config.id2label is not None
# print("Predicted class:", model.config.id2label[predicted_class_idx])