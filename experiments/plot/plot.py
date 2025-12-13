import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

questions = []
with open("experiments/benchmark_generation/questions.jsonl", "r") as f:
    for line in f:
        questions.append(pd.read_json(line, typ="series"))

df = pd.read_csv("experiments/evaluate/model_judgments.csv")

df["difficulty"] = [q["difficulty"] for q in questions]
df["category"] = [q["category"] for q in questions]

difficulties = ["easy", "medium", "hard"]
diff_means = {}
for diff in difficulties:
    subset = df[df["difficulty"] == diff]
    diff_means[diff] = {
        "base": subset["base_score"].mean(),
        "rag": subset["rag_score"].mean(),
        "agentic_rag": subset["agentic_rag_score"].mean(),
    }

overall = {
    "base": df["base_score"].mean(),
    "rag": df["rag_score"].mean(),
    "agentic_rag": df["agentic_rag_score"].mean(),
}
diff_means["overall"] = overall
difficulties.append("overall")
diff_labels = {
    diff: " ".join(part.capitalize() for part in diff.split("_"))
    for diff in difficulties
}

categories = df["category"].unique()
cat_means = {}
for cat in categories:
    subset = df[df["category"] == cat]
    cat_means[cat] = {
        "base": subset["base_score"].mean(),
        "rag": subset["rag_score"].mean(),
        "agentic_rag": subset["agentic_rag_score"].mean(),
    }

cats = sorted(cat_means.keys(), key=lambda c: cat_means[c]["agentic_rag"], reverse=True)

colors = sns.color_palette("Set2", 3)

x = np.arange(len(difficulties))
width = 0.25

fig1, ax1 = plt.subplots(figsize=(10, 6))

bars_base = ax1.bar(
    x - width,
    [diff_means[d]["base"] for d in difficulties],
    width,
    label="Base",
    color=colors[0],
)
bars_rag = ax1.bar(
    x, [diff_means[d]["rag"] for d in difficulties], width, label="RAG", color=colors[1]
)
bars_agentic = ax1.bar(
    x + width,
    [diff_means[d]["agentic_rag"] for d in difficulties],
    width,
    label="Agentic RAG",
    color=colors[2],
)

ax1.set_xlabel("Difficulty")
ax1.set_ylabel("Average Score")
ax1.set_xticks(x)
ax1.set_xticklabels([diff_labels[d] for d in difficulties])
ax1.set_ylim(0, 1.05)
ax1.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
ax1.grid(True, alpha=0.25)

for bars in (bars_base, bars_rag, bars_agentic):
    ax1.bar_label(bars, fmt="%.2f", padding=1, fontsize=12)

fig1.tight_layout()
fig1.savefig("model_performance_difficulty.png", bbox_inches="tight")

cat_labels = {
    cat: " ".join(part.capitalize() for part in cat.split("_")) for cat in cats
}
x2 = np.arange(len(cats))
fig_height = max(6, 0.6 * len(cats))

fig2, ax2 = plt.subplots(figsize=(14, fig_height))

bars_base_c = ax2.bar(
    x2 - width,
    [cat_means[c]["base"] for c in cats],
    width,
    label="Base",
    color=colors[0],
)
bars_rag_c = ax2.bar(
    x2, [cat_means[c]["rag"] for c in cats], width, label="RAG", color=colors[1]
)
bars_agentic_c = ax2.bar(
    x2 + width,
    [cat_means[c]["agentic_rag"] for c in cats],
    width,
    label="Agentic RAG",
    color=colors[2],
)

ax2.set_xlabel("Category")
ax2.set_ylabel("Average Score")
ax2.set_xticks(x2)
ax2.set_xticklabels([cat_labels[c] for c in cats], rotation=30, ha="right")
ax2.set_ylim(0, 1.05)
ax2.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
ax2.grid(True, axis="y", alpha=0.25)

for bars in (bars_base_c, bars_rag_c, bars_agentic_c):
    ax2.bar_label(bars, fmt="%.2f", padding=1, fontsize=12)

fig2.tight_layout()
fig2.savefig("model_performance_categories.png", bbox_inches="tight")
