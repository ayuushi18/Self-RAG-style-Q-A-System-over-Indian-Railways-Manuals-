import json


def read_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


old_j = read_rows("data/judgments_prompt_selfrag.jsonl")
new_j = read_rows("data/judgments_azure_selfrag.jsonl")
old_r = read_rows("data/results_prompt_selfrag.jsonl")
new_r = read_rows("data/results_azure_selfrag.jsonl")

changed_search = sum(old_r[i]["retrieved"] != new_r[i]["retrieved"] for i in old_r)
print("Questions where the retrieved chunks differ:", changed_search, "of", len(old_r))

flipped = [i for i in old_j if old_j[i]["verdict"] != new_j[i]["verdict"]]
print("Questions where the grade changed:", len(flipped))
for i in flipped:
    print("\n-", old_j[i]["question"][:100])
    print("   old search:", old_j[i]["verdict"], "| Azure search:", new_j[i]["verdict"])