import numpy as np  
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import csv

from matplotlib import cm

# Define TPT and C-TPT configurations
lambda_variant_files = [
    ("tuned_lambda_-20.0.csv", "TPT + A-TPT (λ: 70,"),
    ("tuned_lambda_-30.0.csv", "TPT + A-TPT (λ: 75,"),
    ("tuned_lambda_-25.0.csv", "TPT + A-TPT (λ: 80,"),
    ("tuned_lambda_0.0.csv", "TPT (λ: 0,"),
]

colors_shapes = [('purple', 'p'), ('green', '^'), ('brown', 'h'),  ('orange', 's')]

# prompt visualization

features = []
prompt_labels = []
legend_labels = []

for idx, (filename, display_name) in enumerate(lambda_variant_files):
    csv_path = os.path.join("saved_tuned_csv", filename)
    
    if not os.path.exists(csv_path):
        print(f"[Warning] File not found: {csv_path}")
        continue

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        prompt_feats = []
        acc, ece, ad = None, None, None

        for row in reader:
            feature_index = int(row["feature_index"])
            feature_vector = [float(row[f'feature_dim_{i}']) for i in range(len(row)-5)]
            prompt_feats.append(feature_vector)

            acc = float(row['accuracy'])
            ece = float(row['ECE'])
            ad = float(row['AD'])

    features.append(np.array(prompt_feats))
    prompt_labels += [idx] * len(prompt_feats)

    label = f"{display_name} Acc: {acc:.1f}, ECE: {ece:.2f}, AD: {-ad:.4f})"
    legend_labels.append(label)

# Stack and prepare for t-SNE
features = np.vstack(features)
prompt_labels = np.array(prompt_labels)

# Prompt visualization
tsne_prompt = TSNE(n_components=2, perplexity=30, random_state=42, init='random', learning_rate=200.0)
prompt_embeds = tsne_prompt.fit_transform(features)

# Class visualization

num_classes = len(features) // len(lambda_variant_files)
class_ids = np.tile(np.arange(num_classes), len(lambda_variant_files))

tsne_class = TSNE(n_components=2, perplexity=30, random_state=42, init='random', learning_rate=200.0)
class_embeds = tsne_class.fit_transform(features)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Prompt t-SNE plot
for idx, (color, shape) in enumerate(colors_shapes):
    ax1.scatter(prompt_embeds[prompt_labels == idx, 0], 
                prompt_embeds[prompt_labels == idx, 1],
                color=color, marker=shape, label=legend_labels[idx], alpha=0.7, s=60)

# Class t-SNE plot
cmap = cm.get_cmap('jet', num_classes)
for i, class_id in enumerate(class_ids):
    ax2.scatter(class_embeds[i, 0], class_embeds[i, 1],
                color=cmap(class_id), marker='o', s=40)

# Axis titles
ax1.text(0.5, -0.1, 'Prompt Visualization', fontsize=24, weight='bold',
         ha='center', transform=ax1.transAxes)
ax2.text(0.5, -0.1, 'Class Visualization', fontsize=24, weight='bold',
         ha='center', transform=ax2.transAxes)

# Legend on top
fig.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.26),
           fontsize=24, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("tpt_ctpt_visualization.png", bbox_inches='tight', pad_inches=0.1, transparent=False, dpi=300)