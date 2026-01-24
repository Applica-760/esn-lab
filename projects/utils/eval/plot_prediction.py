import numpy as np
import matplotlib.pyplot as plt


def plot_prediction(results: list, target_id: str, save_path: str, ext: str = "png"):
    """
    resultのリストからidのデータを取り出しプロットする
    """
    target_data = None
    for result in results:
        if result["id"] == target_id:
            target_data = result
            break
    
    if target_data is None:
        raise ValueError(f"ID '{target_id}' not found in results")
    
    predictions = np.array(target_data["predictions"])  # shape: [時刻, 3]
    labels = np.array(target_data["labels"])  # shape: [時刻, 3] (one-hot形式)
    
    pred_indices = np.argmax(predictions, axis=1) 
    label_indices = np.argmax(labels, axis=1)  
    
    # predictions/labels: other=0, foraging=1, rumination=2
    # プロットy軸: foraging=3, rumination=2, other=1
    index_to_y = {0: 1, 1: 3, 2: 2}  # other->1, foraging->3, rumination->2
    
    pred_y = np.array([index_to_y[idx] for idx in pred_indices])
    label_y = np.array([index_to_y[idx] for idx in label_indices])
    time_axis = np.arange(len(predictions))
    
    total = len(pred_indices)
    foraging_ratio = np.sum(pred_indices == 1) / total * 100
    rumination_ratio = np.sum(pred_indices == 2) / total * 100
    other_ratio = np.sum(pred_indices == 0) / total * 100
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, label_y, color="black", linewidth=2.0, label="Label")
    ax.scatter(time_axis, pred_y, color="darkorange", s=20, alpha=0.3, label="Prediction")
    
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["other", "rumination", "foraging"])
    ax.set_ylim(0.5, 3.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Behavior")
    ax.set_title(f"Prediction & Label (ID: {target_id})")
    
    ratio_text = f"foraging: {foraging_ratio:.1f}%, rumination: {rumination_ratio:.1f}%, other: {other_ratio:.1f}%"
    ax.legend(loc="upper right", title=ratio_text)
    
    plt.tight_layout()
    save_file = f"{save_path}.{ext}"
    plt.savefig(save_file, dpi=150)
    plt.close()