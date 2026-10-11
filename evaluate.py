import argparse
from tqdm import tqdm
from datasets import load_dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from model import HazardPredictionModel
import torch


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def evaluate_model(dataset_name="DHPR/Driving-Hazard-Prediction-and-Reasoning", split="validation", max_samples=None, device="cpu"):
    print(f"Loading dataset {dataset_name} ({split})...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print("Initializing HazardPredictionModel...")
    model = HazardPredictionModel(device=device)

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    smoothie = SmoothingFunction().method4

    print("Starting evaluation...")
    for item in tqdm(dataset):
        
        image = item.get('image')
        reference = item.get('hazard', item.get('text', ''))
        
        if image is None or not reference:
            continue
            
        generated_text, _ = model.generate(image)
        
        
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(generated_text.lower())
        
        
        b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        b2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        b3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        b4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)

    print("\n--- Evaluation Results ---")
    print(f"Evaluated Samples: {len(bleu4_scores)}")
    print(f"BLEU-1: {sum(bleu1_scores)/len(bleu1_scores):.4f}")
    print(f"BLEU-2: {sum(bleu2_scores)/len(bleu2_scores):.4f}")
    print(f"BLEU-3: {sum(bleu3_scores)/len(bleu3_scores):.4f}")
    print(f"BLEU-4: {sum(bleu4_scores)/len(bleu4_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hazard Prediction Model")
    parser.add_argument("--dataset", type=str, default="DHPR/Driving-Hazard-Prediction-and-Reasoning", help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    evaluate_model(args.dataset, args.split, args.max_samples, args.device)
