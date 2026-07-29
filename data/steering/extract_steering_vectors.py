"""
Extract steering vectors for Llama-3.2-3B at layers 8, 16, and 24.

For each concept, computes: mean(positive_activations) - mean(negative_activations)
at each target layer using mean-pooled residual stream activations.

Output: data/steering/vectors-llama3b.json  (15 vectors: 5 concepts × 3 layers)
"""

import json
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
TARGET_LAYERS = [8, 16, 24]
HIDDEN_SIZE = 3072
OUT_PATH = Path(__file__).parent / "vectors-llama3b.json"

# ── Concept example pairs ─────────────────────────────────────────────────────
# positive = examples that exhibit the concept
# negative = contrasting examples without the concept

CONCEPTS = {
    "positive_sentiment": {
        "positive": [
            "I absolutely love this! It's the best thing I've ever experienced.",
            "This is wonderful and I'm so happy with the results.",
            "What an amazing day! Everything went perfectly.",
            "I'm thrilled with this purchase. Highly recommend to everyone!",
            "The service was exceptional and the staff were incredibly kind.",
            "This made my day! I feel so grateful and joyful.",
            "Fantastic work! I couldn't be more impressed with the outcome.",
            "Life is beautiful and I'm filled with happiness and appreciation.",
            "What a delightful experience! I'll definitely be back for more.",
            "I'm overjoyed with this result. Simply perfect in every way!",
        ],
        "negative": [
            "This is absolutely terrible. I'm deeply disappointed and upset.",
            "What a waste of money. I regret making this purchase.",
            "The worst experience I've ever had. Completely awful.",
            "I'm furious with the poor service and terrible quality.",
            "This ruined my entire day. I am miserable and frustrated.",
            "Horrible. Just horrible. I can't believe how bad this is.",
            "Disgusting and infuriating. I want a refund immediately.",
            "I hate everything about this. It was a total disaster.",
            "The most frustrating and stressful thing I've ever dealt with.",
            "Dreadful quality and terrible service. Avoid at all costs.",
        ],
    },
    "negative_sentiment": {
        "positive": [
            "This is absolutely terrible. I'm deeply disappointed and upset.",
            "What a waste of money. I regret making this purchase.",
            "The worst experience I've ever had. Completely awful.",
            "I'm furious with the poor service and terrible quality.",
            "This ruined my entire day. I am miserable and frustrated.",
            "Horrible. Just horrible. I can't believe how bad this is.",
            "Disgusting and infuriating. I want a refund immediately.",
            "I hate everything about this. It was a total disaster.",
            "The most frustrating and stressful thing I've ever dealt with.",
            "Dreadful quality and terrible service. Avoid at all costs.",
        ],
        "negative": [
            "I absolutely love this! It's the best thing I've ever experienced.",
            "This is wonderful and I'm so happy with the results.",
            "What an amazing day! Everything went perfectly.",
            "I'm thrilled with this purchase. Highly recommend to everyone!",
            "The service was exceptional and the staff were incredibly kind.",
            "This made my day! I feel so grateful and joyful.",
            "Fantastic work! I couldn't be more impressed with the outcome.",
            "Life is beautiful and I'm filled with happiness and appreciation.",
            "What a delightful experience! I'll definitely be back for more.",
            "I'm overjoyed with this result. Simply perfect in every way!",
        ],
    },
    "french_language": {
        "positive": [
            "Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien pour vous.",
            "Je suis très heureux de vous rencontrer. C'est un vrai plaisir.",
            "La France est un beau pays avec une riche culture et une longue histoire.",
            "Voulez-vous manger quelque chose? Il y a un bon restaurant tout près d'ici.",
            "Je parle français depuis mon enfance. C'est ma langue maternelle.",
            "Le soleil brille aujourd'hui et le temps est absolument magnifique.",
            "Mes amis et moi aimons nous retrouver pour discuter de tout et de rien.",
            "L'art et la littérature françaises sont reconnus et admirés dans le monde entier.",
            "Quelle belle journée pour se promener dans le parc avec toute la famille.",
            "Je voudrais commander un café crème et un croissant, s'il vous plaît.",
        ],
        "negative": [
            "Hello, how are you doing today? I hope everything is going well for you.",
            "I am very happy to meet you. It's truly a pleasure to make your acquaintance.",
            "The country has a rich culture and a long and fascinating history.",
            "Would you like to eat something? There is a great restaurant just nearby.",
            "I have been speaking English since childhood. It is my native language.",
            "The sun is shining today and the weather is absolutely beautiful outside.",
            "My friends and I enjoy getting together to talk about everything and anything.",
            "Art and literature from this region are recognized and admired around the world.",
            "What a beautiful day to take a walk in the park with the whole family.",
            "I would like to order a coffee and a pastry, please.",
        ],
    },
    "python_coding": {
        "positive": [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "import numpy as np\ndata = np.array([1, 2, 3, 4, 5])\nmean = np.mean(data)\nprint(f'Mean: {mean}')",
            "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()",
            "with open('data.csv', 'r') as f:\n    lines = f.readlines()\nfor line in lines:\n    print(line.strip().split(','))",
            "from typing import List, Dict\ndef process(items: List[Dict]) -> List[str]:\n    return [item['name'] for item in items if item.get('active')]",
            "import torch\nmodel = torch.nn.Linear(128, 64)\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn = torch.nn.MSELoss()",
            "result = [x**2 for x in range(10) if x % 2 == 0]\nprint(result)  # [0, 4, 16, 36, 64]",
            "try:\n    value = int(input('Enter a number: '))\nexcept ValueError as e:\n    print(f'Invalid input: {e}')",
            "import json\nwith open('config.json') as f:\n    config = json.load(f)\nmodel_path = config.get('model_path', '/default/path')",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        ],
        "negative": [
            "The recipe calls for two cups of flour, one egg, and a pinch of salt to taste.",
            "The meeting is scheduled for Thursday at three in the afternoon in the main conference room.",
            "The history of ancient Rome spans over a thousand years of rich civilization and culture.",
            "Please remember to water the houseplants twice a week during the hot summer months.",
            "The new museum exhibit on impressionism opens next weekend and tickets are available online.",
            "Scientists have recently discovered a new species of tropical bird deep in the Amazon rainforest.",
            "The basketball team practiced every single day to prepare for the upcoming championship game.",
            "The restaurant serves authentic Italian cuisine made with fresh, locally sourced ingredients daily.",
            "During our morning hike we saw wildflowers and several deer grazing in the sunny meadow.",
            "The novel explores themes of friendship, loss, identity, and the meaning of finding a home.",
        ],
    },
    "refusal_behavior": {
        "positive": [
            "I'm sorry, but I can't help with that request. It goes against my guidelines.",
            "I'm not able to assist with creating harmful content. I must respectfully decline this request.",
            "That's not something I'm able to help with. I'd be happy to assist with something else.",
            "I need to decline this request as fulfilling it could cause harm to others.",
            "I apologize, but providing that information isn't something I'm able to do for safety reasons.",
            "This falls outside what I'm able to assist with. Is there something else I can help you with?",
            "I'm unable to fulfill this request. It would violate my safety guidelines and values.",
            "I must respectfully decline to help with this particular task as it could be harmful.",
            "That request isn't something I can assist with for important ethical and safety reasons.",
            "I can't help with that. However, if you have other questions I'm here to assist.",
        ],
        "negative": [
            "Sure! I'd be happy to help you with that. Here's everything you need to know.",
            "Of course! Let me walk you through this step by step with a clear explanation.",
            "Absolutely! That's a great question. Here's a detailed and thorough explanation.",
            "Happy to help! Here's everything you need to know about this fascinating topic.",
            "Certainly! I'll explain this as clearly and completely as possible for you.",
            "Great question! Let me provide a comprehensive and helpful answer for you right now.",
            "Of course, I can help with that. Here's exactly what I recommend you do.",
            "Sure thing! This is actually quite straightforward. Let me explain how it works.",
            "I'd love to help with this! Here's my best and most complete answer to your question.",
            "Definitely! Here's a thorough and detailed explanation of exactly how this works.",
        ],
    },
}


def extract_at_layers(
    text: str,
    llama_model,
    tok,
    target_layers: list,
    fa_mask_fn,
) -> dict[int, np.ndarray]:
    """Return mean-pooled residual stream at each target layer."""
    ids = tok.encode(text, add_special_tokens=True)
    if len(ids) > 256:
        ids = ids[:256]

    inp = mx.array([ids])                   # [1, seq_len]
    h = llama_model.embed_tokens(inp)

    fa_mask = fa_mask_fn(h)
    max_layer = max(target_layers)
    target_set = set(target_layers)
    results = {}

    for i, layer in enumerate(llama_model.layers):
        h = layer(h, fa_mask, cache=None)
        if i in target_set:
            h32 = h[0].astype(mx.float32)
            mx.eval(h32)
            arr = np.array(h32)             # [seq_len, hidden]
            results[i] = arr.mean(axis=0)  # mean pool → [hidden]
        if i >= max_layer:
            break

    return results


def compute_steering_vector(
    pos_examples: list[str],
    neg_examples: list[str],
    llama_model,
    tok,
    target_layers: list,
    fa_mask_fn,
) -> dict[int, np.ndarray]:
    """Compute mean(positive) - mean(negative) at each layer."""
    pos_acts = {layer: [] for layer in target_layers}
    neg_acts = {layer: [] for layer in target_layers}

    for text in pos_examples:
        acts = extract_at_layers(text, llama_model, tok, target_layers, fa_mask_fn)
        for layer, vec in acts.items():
            pos_acts[layer].append(vec)

    for text in neg_examples:
        acts = extract_at_layers(text, llama_model, tok, target_layers, fa_mask_fn)
        for layer, vec in acts.items():
            neg_acts[layer].append(vec)

    return {
        layer: np.stack(pos_acts[layer]).mean(axis=0)
               - np.stack(neg_acts[layer]).mean(axis=0)
        for layer in target_layers
    }


def main():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Loaded in {time.time()-t0:.1f}s")

    llama_model = model.model
    num_layers = len(llama_model.layers)
    print(f"Model has {num_layers} layers; extracting at layers {TARGET_LAYERS}")
    assert all(l < num_layers for l in TARGET_LAYERS), "Layer index out of range"

    # Build mask function — swa_idx is None for 3B so always use full causal mask
    def fa_mask_fn(h):
        return create_attention_mask(h, None)

    output = {
        "model": MODEL_ID,
        "num_layers": num_layers,
        "hidden_size": HIDDEN_SIZE,
        "target_layers": TARGET_LAYERS,
        "pooling": "mean_across_tokens",
        "method": "mean_positive_minus_mean_negative",
        "vectors": {},
    }

    total_examples = sum(
        len(v["positive"]) + len(v["negative"]) for v in CONCEPTS.values()
    )
    done = 0
    t_start = time.time()

    for concept_name, examples in CONCEPTS.items():
        print(f"\n[{concept_name}]  "
              f"({len(examples['positive'])} pos, {len(examples['negative'])} neg)")

        steering = compute_steering_vector(
            pos_examples=examples["positive"],
            neg_examples=examples["negative"],
            llama_model=llama_model,
            tok=tokenizer,
            target_layers=TARGET_LAYERS,
            fa_mask_fn=fa_mask_fn,
        )

        output["vectors"][concept_name] = {}
        for layer, vec in steering.items():
            key = f"layer_{layer}"
            output["vectors"][concept_name][key] = vec.tolist()
            norm = float(np.linalg.norm(vec))
            print(f"  layer {layer:2d}: norm={norm:.4f}")

        done += len(examples["positive"]) + len(examples["negative"])
        elapsed = time.time() - t_start
        remaining = total_examples - done
        rate = done / elapsed if elapsed > 0 else 1
        print(f"  ({done}/{total_examples} examples done, "
              f"~{remaining/rate:.0f}s remaining)")

    output["elapsed_seconds"] = round(time.time() - t_start, 1)

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {len(CONCEPTS)} concepts × {len(TARGET_LAYERS)} layers "
          f"= {len(CONCEPTS)*len(TARGET_LAYERS)} vectors to:\n  {OUT_PATH}")
    print(f"File size: {OUT_PATH.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
