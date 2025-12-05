from datasets import load_dataset

def save_boolq_prompts(limit=100, output_file="prompts.txt"):
    print("Downloading original BoolQ dataset (very small)...")
    ds = load_dataset("boolq", split="train")

    prompts = []
    for i in range(min(limit, len(ds))):
        question = ds[i]["question"].strip()
        passage = ds[i]["passage"].strip()

        # Format prompt similar to FLAN templates
        prompt = f"Question: {question}\nPassage: {passage}\nAnswer:"
        prompts.append(prompt)

    with open(output_file, "w", encoding="utf8") as f:
        for p in prompts:
            f.write(p + "\n\n")

    print(f"Saved {len(prompts)} prompts to {output_file}")


if __name__ == "__main__":
    save_boolq_prompts()
