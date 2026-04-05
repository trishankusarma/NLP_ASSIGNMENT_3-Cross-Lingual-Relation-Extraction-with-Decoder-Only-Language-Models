def plot_metrics(train_losses, train_accs, val_f1_score_micro, val_f1_score_macro, output_dir, log_interval=100):
    
    step_axis = [i * log_interval for i in range(1, len(train_losses) + 1)]
    epoch_axis = range(1, len(val_f1_score_micro) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Metrics", fontsize=14)

    # Plot 1: Train Loss (per step → smooth with rolling average)
    smoothed_loss = []
    window = 5  # rolling average over 5 log points
    for i in range(len(train_losses)):
        start = max(0, i - window + 1)
        smoothed_loss.append(sum(train_losses[start:i+1]) / (i - start + 1))

    axes[0].plot(step_axis, train_losses, 'b-', alpha=0.3, linewidth=1, label="Raw")
    axes[0].plot(step_axis, smoothed_loss, 'b-', linewidth=2, label="Smoothed")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Train Accuracy (per step → smooth)
    smoothed_acc = []
    for i in range(len(train_accs)):
        start = max(0, i - window + 1)
        smoothed_acc.append(sum(train_accs[start:i+1]) / (i - start + 1))

    axes[1].plot(step_axis, train_accs, 'g-', alpha=0.3, linewidth=1, label="Raw")
    axes[1].plot(step_axis, smoothed_acc, 'g-', linewidth=2, label="Smoothed")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Val F1 per epoch
    axes[2].plot(epoch_axis, val_f1_score_micro, 'r-o', linewidth=2, markersize=6, label="Micro F1")
    axes[2].plot(epoch_axis, val_f1_score_macro, 'm-o', linewidth=2, markersize=6, label="Macro F1")
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir}/training_metrics.png")