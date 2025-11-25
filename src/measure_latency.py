import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--device", default="cpu") 
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    

    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    times_ms = []
    
    # Warmup
    for _ in range(10):
        enc = tokenizer(texts[0], truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            _ = model(**enc)

    print(f"Measuring latency over {args.runs} runs...")
    for i in range(args.runs):
        text = texts[i % len(texts)]
        enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**enc)
        end = time.perf_counter()
        
        times_ms.append((end - start) * 1000.0)

    p95 = sorted(times_ms)[int(0.95 * len(times_ms))]
    print(f"P95 Latency: {p95:.2f} ms")

if __name__ == "__main__":
    main()