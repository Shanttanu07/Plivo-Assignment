import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu") 
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    

    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    model.eval()
    model.to(args.device)

    results = {}
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"Running prediction on {len(data)} examples...")

    for obj in data:
        text = obj["text"]
        
 
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, return_tensors="pt")
        
 
        offsets = enc.pop("offset_mapping")[0].tolist()
        
    
        inputs = {k: v.to(args.device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        
        preds = logits.argmax(dim=-1).tolist()
        
   
        ents = []
        for (start, end), pid in zip(offsets, preds):
            if start == end: continue # Skip special tokens
            
            label = ID2LABEL.get(pid, "O")
            if label.startswith("B-") or label.startswith("I-"):
                clean_label = label[2:]
                ents.append({
                    "start": start, "end": end, 
                    "label": clean_label, 
                    "pii": label_is_pii(clean_label)
                })
        
 
        merged = []
        if ents:
            curr = ents[0]
            for next_ent in ents[1:]:

                gap = next_ent['start'] - curr['end']
                

                if next_ent['label'] == curr['label'] and gap <= 2:
                    curr['end'] = next_ent['end'] 
                else:
                    merged.append(curr)
                    curr = next_ent
            merged.append(curr)

        results[obj["id"]] = merged

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()