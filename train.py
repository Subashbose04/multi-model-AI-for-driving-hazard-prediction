import argparse
import os
from PIL import Image, ImageDraw

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, DatasetDict
import evaluate



def generate_bounding_boxes(image: Image.Image, detector=None, threshold: float = 0.5):
    """Run an object detector to propose bounding boxes.

    We use a pretrained torchvision detector by default.  The
    DHPR paper included boxes for validation, but when running
the model on a single image we can ask a detector to highlight
regions of interest (hazards).  In practice you would replace
this with a dedicated hazard proposal network or annotate the
dataset manually.

    Returns a list of (x1, y1, x2, y2) coordinates.
    """
    if detector is None:
        # load a quick off-the-shelf Faster R-CNN
        detector = torch.hub.load(
            "pytorch/vision:v0.14.0", "fasterrcnn_resnet50_fpn", pretrained=True
        ).eval()
    transform = transforms = torch.nn.Sequential()
    # convert to tensor inside detector call

    img_tensor = transforms = None
    # torchvision detectors accept PIL images when scripted, so simply:
    outputs = detector([image])
    boxes = []
    if len(outputs) > 0:
        for score, box in zip(outputs[0]["scores"], outputs[0]["boxes"]):
            if score >= threshold:
                boxes.append(box.tolist())
    return boxes


def draw_boxes(image: Image.Image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    return image

# ------------------------------------------------------------
# data preparation
# ------------------------------------------------------------


def preprocess_examples(examples, processor):
    # the dataset is expected to have an "image" column that is a path or
    # PIL image and a "caption" column with the hazard description.
    images = [Image.open(p).convert("RGB") for p in examples["image"]]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    # tokenize captions
    captions = examples.get("caption", None)
    if captions is not None:
        labels = processor.tokenizer(captions, padding="max_length", truncation=True, max_length=32).input_ids
        inputs["labels"] = labels
    return inputs

# ------------------------------------------------------------
# training / evaluation logic
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train or run hazard-prediction model")
    parser.add_argument("--dataset", type=str, default="dhpr", help="Hugging Face dataset identifier or local path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save checkpoints")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_infer", action="store_true")
    parser.add_argument("--image", type=str, help="Single image file for inference")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # load model & processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-base")

    # load dataset
    if os.path.exists(args.dataset):
        dataset = load_dataset("imagefolder", data_dir=args.dataset)
        # assume folder structure with subfolders or a csv specifying captions
    else:
        dataset = load_dataset(args.dataset)

    # if dataset doesn't already have splits, split ourselves
    if isinstance(dataset, DatasetDict) is False:
        dataset = dataset.train_test_split(test_size=0.1)
    if "validation" not in dataset:
        dataset["validation"] = dataset["test"]

    # preprocess (map dynamically to avoid storing pixel_values in dataset)
    def _map_fn(examples):
        return preprocess_examples(examples, processor)

    tokenized = dataset.map(_map_fn, batched=True, remove_columns=dataset["train"].column_names)

    # BLEU metric for eval
    bleu = evaluate.load("bleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        # bleu returns a dict with bleu, maybe not bleu-4 only but that's fine
        return {"bleu": result["bleu"]}

    # training
    if args.do_train:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=100,
            num_train_epochs=3,
            warmup_steps=500,
            save_total_limit=2,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=processor.tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    if args.do_eval:
        trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=processor.tokenizer)
        metrics = trainer.evaluate(eval_dataset=tokenized["validation"], metric_key_prefix="eval")
        print(metrics)

    if args.do_infer:
        if not args.image:
            raise ValueError("--image must be provided for inference")
        from torchvision import transforms
        pil_image = Image.open(args.image).convert("RGB")
        # generate bounding boxes with a pretrained detector for demonstration
        try:
            import torchvision
            detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            detector.eval()
        except ImportError:
            detector = None
        boxes = generate_bounding_boxes(pil_image, detector=detector, threshold=0.7)
        out_img = draw_boxes(pil_image.copy(), boxes)
        out_img.save("boxed.png")
        # now generate caption
        inputs = processor(images=pil_image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        print("Predicted hazard description:\n", caption)

if __name__ == "__main__":
    main()
