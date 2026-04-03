import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from tqdm import tqdm

def find_subsequence(sequence, subseq):
    n, m = len(sequence), len(subseq)
    for i in range(n - m + 1):
        if sequence[i:i + m] == subseq:
            return i
    return -1

class MessageDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, device):
    input_ids_list = []
    target_ids_list = []
    start_pos_list = []
    
    for item in batch:
        input_ids = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )[0]
        
        target_ids = tokenizer(
            item["response"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        
        start_pos = find_subsequence(input_ids.tolist(), target_ids.tolist())
        if start_pos == -1:
            continue
            
        input_ids_list.append(input_ids)
        target_ids_list.append(target_ids)
        start_pos_list.append(start_pos)
    
    if not input_ids_list:
        return None
    
    # Pad sequences
    max_len = max(len(x) for x in input_ids_list)
    padded_input_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        padded_input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1
    
    return {
        "input_ids": padded_input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "target_ids_list": target_ids_list,
        "start_pos_list": start_pos_list,
    }

def compute_logps_batch(model, batch_data):
    with torch.no_grad():
        logits = model(
            input_ids=batch_data["input_ids"],
            attention_mask=batch_data["attention_mask"]
        ).logits
    
    results = []
    for i, (target_ids, start_pos) in enumerate(zip(batch_data["target_ids_list"], batch_data["start_pos_list"])):
        token_logps = []
        for j in range(len(target_ids)):
            token_pos = start_pos + j
            if token_pos == 0:
                continue
            prev_logits = logits[i, token_pos - 1]
            log_probs = F.log_softmax(prev_logits, dim=-1)
            token_logps.append(log_probs[target_ids[j]].item())
        
        results.append({
            "token_logps": token_logps,
            "joint_logp": sum(token_logps),
        })
    
    return results

def load_model(model_path, device_ids):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids)
    
    model = model.to(f"cuda:{device_ids[0]}")
    model.eval()
    
    return tokenizer, model

def process_data(model, tokenizer, data, batch_size, device):
    dataset = MessageDataset(data, tokenizer)
    results = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch_data = collate_fn(batch, tokenizer, device)
        
        if batch_data is None:
            continue
        
        batch_results = compute_logps_batch(model, batch_data)
        results.extend(batch_results)
    
    return results

def compare_models(base_path, trained_path, data_path, output_path, batch_size, device_ids):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Prepare data
    print(f"Preparing {len(test_data)} samples...")
    processed_data = []
    for ele in tqdm(test_data, desc="Preparing data"):
        if "response" not in ele:
            ele["response"] = json.dumps(ele["resp_model"], ensure_ascii=False)
        ele["messages"].append({"role": "assistant", "content": ele["response"]})
        processed_data.append(ele)
    
    device = f"cuda:{device_ids[0]}"
    
    # Process base model
    print("Loading base model...")
    tokenizer, base_model = load_model(base_path, device_ids)
    print("Processing base model...")
    base_results = process_data(base_model, tokenizer, processed_data, batch_size, device)
    del base_model
    torch.cuda.empty_cache()
    
    # Process trained model
    print("Loading trained model...")
    tokenizer, trained_model = load_model(trained_path, device_ids)
    print("Processing trained model...")
    trained_results = process_data(trained_model, tokenizer, processed_data, batch_size, device)
    del trained_model
    torch.cuda.empty_cache()
    
    # Compare results
    print("Comparing results...")
    comparisons = []
    for i, (base_res, trained_res) in enumerate(tqdm(zip(base_results, trained_results), total=len(base_results), desc="Computing differences")):
        token_diffs = [t - b for b, t in zip(base_res["token_logps"], trained_res["token_logps"])]
        comparisons.append({
            "index": i,
            "base_joint_logp": base_res["joint_logp"],
            "trained_joint_logp": trained_res["joint_logp"],
            "joint_logp_diff": trained_res["joint_logp"] - base_res["joint_logp"],
            "base_token_logps": base_res["token_logps"],
            "trained_token_logps": trained_res["token_logps"],
            "token_logp_diffs": token_diffs,
        })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    joint_diffs = [c["joint_logp_diff"] for c in comparisons]
    print(f"\nProcessed {len(comparisons)} samples")
    print(f"Joint logp diff - Mean: {np.mean(joint_diffs):.4f}, Std: {np.std(joint_diffs):.4f}")
    print(f"Joint logp diff - Min: {np.min(joint_diffs):.4f}, Max: {np.max(joint_diffs):.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--trained_model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save comparison results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device_ids", type=str, default="0", help="Comma-separated GPU IDs, e.g., '0,1,2,3'")
    
    args = parser.parse_args()
    device_ids = [int(x) for x in args.device_ids.split(",")]
    
    compare_models(
        args.base_model,
        args.trained_model,
        args.data_path,
        args.output_path,
        args.batch_size,
        device_ids
    )
