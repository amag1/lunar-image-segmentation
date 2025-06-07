import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

result_dir = "dataset/images/resulting_masks"
output_dir = "metrics_output"
os.makedirs(output_dir, exist_ok=True)

loss_pattern = re.compile(r"loss(\d+)\.png$")

losses_per_class = {}
all_losses = []

for class_name in os.listdir(result_dir):
    class_path = os.path.join(result_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    losses = []
    for fname in os.listdir(class_path):
        m = loss_pattern.search(fname)
        if m:
            loss = int(m.group(1))
            precision = 100 - loss
            losses.append(precision)
            all_losses.append(precision)

    losses_per_class[class_name] = np.array(losses)


def freedman_diaconis_bins(data):
    if data.size < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else None
    if not bin_width or bin_width == 0:
        return int(np.sqrt(len(data)))
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(bins, 1)


stats = []

for cls, vals in losses_per_class.items():
    if vals.size == 0:
        stats.append([cls, None, None, None, None])
        continue
    stats.append([cls, vals.min(), vals.max(), round(vals.mean(), 2), np.median(vals)])

# Overall stats
all_losses_arr = np.array(all_losses)
stats.append(
    [
        "Overall",
        all_losses_arr.min() if all_losses_arr.size else None,
        all_losses_arr.max() if all_losses_arr.size else None,
        round(all_losses_arr.mean(), 2) if all_losses_arr.size else None,
        np.median(all_losses_arr) if all_losses_arr.size else None,
    ]
)

# Save stats to CSV
csv_path = os.path.join(output_dir, "loss_stats.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Min", "Max", "Average", "Median"])
    writer.writerows(stats)


# Save histograms
def save_histogram(data, name):
    if data.size == 0:
        return
    bins = freedman_diaconis_bins(data)
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.7)
    plt.title(f"Distribución de la función de pérdida (precisión) - {name}")
    plt.xlabel("Precisión (%)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_distribution_{name}.png"))
    plt.close()


for cls, vals in losses_per_class.items():
    save_histogram(vals, cls)

save_histogram(all_losses_arr, "Total")
