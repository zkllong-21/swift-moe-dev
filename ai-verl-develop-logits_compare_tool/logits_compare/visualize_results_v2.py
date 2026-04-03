import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(results_path, output_dir):
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    joint_diffs = [d['joint_logp_diff'] for d in data]
    base_joints = [d['base_joint_logp'] for d in data]
    trained_joints = [d['trained_joint_logp'] for d in data]
    
    # Token-level diffs (flatten all tokens)
    all_token_diffs = []
    for d in data:
        all_token_diffs.extend(d['token_logp_diffs'])
    
    # Filtered token diffs (|diff| >= 0.1)
    filtered_token_diffs = [d for d in all_token_diffs if abs(d) >= 0.1]
    
    # 1. Sample-level joint logp comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot
    axes[0, 0].scatter(base_joints, trained_joints, alpha=0.5, s=20)
    axes[0, 0].plot([min(base_joints), max(base_joints)], [min(base_joints), max(base_joints)], 'r--', lw=2)
    axes[0, 0].set_xlabel('Base Model Joint LogP')
    axes[0, 0].set_ylabel('Trained Model Joint LogP')
    axes[0, 0].set_title('Sample-level Joint LogP Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Joint logp diff distribution
    axes[0, 1].hist(joint_diffs, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].axvline(np.mean(joint_diffs), color='g', linestyle='--', lw=2, label=f'Mean: {np.mean(joint_diffs):.3f}')
    axes[0, 1].set_xlabel('Joint LogP Difference (Trained - Base)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sample-level Joint LogP Difference Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot
    axes[1, 0].boxplot([base_joints, trained_joints], labels=['Base', 'Trained'])
    axes[1, 0].set_ylabel('Joint LogP')
    axes[1, 0].set_title('Sample-level Joint LogP Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_diffs = np.sort(joint_diffs)
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    axes[1, 1].plot(sorted_diffs, cumulative, lw=2)
    axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Joint LogP Difference')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution of Joint LogP Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_level_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sample_level_analysis.png'}")
    plt.close()
    
    # 2. Token-level analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Token diff distribution
    axes[0, 0].hist(all_token_diffs, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[0, 0].axvline(np.mean(all_token_diffs), color='g', linestyle='--', lw=2, label=f'Mean: {np.mean(all_token_diffs):.3f}')
    axes[0, 0].set_xlabel('Token LogP Difference (Trained - Base)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Token-level LogP Difference Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Token position analysis (first 50 samples)
    sample_indices = list(range(min(50, len(data))))
    for idx in sample_indices[:10]:
        axes[0, 1].plot(data[idx]['token_logp_diffs'], alpha=0.3, lw=1)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Token Position')
    axes[0, 1].set_ylabel('LogP Difference')
    axes[0, 1].set_title('Token-level LogP Difference by Position (10 samples)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Violin plot
    parts = axes[1, 0].violinplot([all_token_diffs], positions=[0], showmeans=True, showmedians=True)
    axes[1, 0].set_ylabel('Token LogP Difference')
    axes[1, 0].set_title('Token-level LogP Difference Violin Plot')
    axes[1, 0].set_xticks([0])
    axes[1, 0].set_xticklabels(['All Tokens'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(all_token_diffs, percentiles)
    axes[1, 1].bar(range(len(percentiles)), values, tick_label=[f'P{p}' for p in percentiles])
    axes[1, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Percentile')
    axes[1, 1].set_ylabel('Token LogP Difference')
    axes[1, 1].set_title('Token-level LogP Difference Percentiles')
    axes[1, 1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'token_level_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'token_level_analysis.png'}")
    plt.close()
    
    # 3. Heatmap for sample-token matrix (first 50 samples, max 100 tokens)
    max_samples = min(50, len(data))
    max_tokens = 100
    matrix = []
    for i in range(max_samples):
        row = data[i]['token_logp_diffs'][:max_tokens]
        if len(row) < max_tokens:
            row = row + [0] * (max_tokens - len(row))
        matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Sample Index')
    ax.set_title('Token LogP Difference Heatmap (First 50 Samples)')
    plt.colorbar(im, ax=ax, label='LogP Difference')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'heatmap_analysis.png'}")
    plt.close()
    
    # 4. Analyze token positions with changes
    # Build position-level statistics
    position_diffs = {}
    for d in data:
        for pos, diff in enumerate(d['token_logp_diffs']):
            if pos not in position_diffs:
                position_diffs[pos] = []
            position_diffs[pos].append(diff)
    
    # Calculate mean diff for all positions
    all_positions = []
    for pos, diffs in position_diffs.items():
        mean_diff = np.mean(diffs)
        all_positions.append({
            "position": pos,
            "mean_diff": float(mean_diff),
            "sample_count": len(diffs),
            "sample_ratio": float(len(diffs) / len(data))
        })
    
    # Sort and get top 20 for decrease and increase
    decrease_positions = [p for p in all_positions if p['mean_diff'] < 0]
    increase_positions = [p for p in all_positions if p['mean_diff'] > 0]
    
    decrease_positions.sort(key=lambda x: x['mean_diff'])
    increase_positions.sort(key=lambda x: x['mean_diff'], reverse=True)
    
    top_decrease_positions = decrease_positions[:20]
    top_increase_positions = increase_positions[:20]
    
    # 5. Statistics summary
    stats = {
        "sample_level": {
            "total_samples": len(data),
            "joint_logp_diff": {
                "mean": float(np.mean(joint_diffs)),
                "std": float(np.std(joint_diffs)),
                "min": float(np.min(joint_diffs)),
                "max": float(np.max(joint_diffs)),
                "median": float(np.median(joint_diffs)),
                "improved_samples": int(sum(1 for d in joint_diffs if d > 0)),
                "degraded_samples": int(sum(1 for d in joint_diffs if d < 0)),
            }
        },
        "token_level": {
            "all_tokens": {
                "total_tokens": len(all_token_diffs),
                "logp_diff": {
                    "mean": float(np.mean(all_token_diffs)),
                    "std": float(np.std(all_token_diffs)),
                    "min": float(np.min(all_token_diffs)),
                    "max": float(np.max(all_token_diffs)),
                    "median": float(np.median(all_token_diffs)),
                    "improved_tokens": int(sum(1 for d in all_token_diffs if d > 0)),
                    "degraded_tokens": int(sum(1 for d in all_token_diffs if d < 0)),
                }
            },
            "filtered_tokens_abs_diff_gte_0.1": {
                "total_tokens": len(filtered_token_diffs),
                "logp_diff": {
                    "mean": float(np.mean(filtered_token_diffs)) if filtered_token_diffs else 0.0,
                    "std": float(np.std(filtered_token_diffs)) if filtered_token_diffs else 0.0,
                    "min": float(np.min(filtered_token_diffs)) if filtered_token_diffs else 0.0,
                    "max": float(np.max(filtered_token_diffs)) if filtered_token_diffs else 0.0,
                    "median": float(np.median(filtered_token_diffs)) if filtered_token_diffs else 0.0,
                    "improved_tokens": int(sum(1 for d in filtered_token_diffs if d > 0)),
                    "degraded_tokens": int(sum(1 for d in filtered_token_diffs if d < 0)),
                }
            }
        },
        "position_level_analysis": {
            "top_decrease_positions": top_decrease_positions,
            "top_increase_positions": top_increase_positions,
            "description": "Top 20 token positions with largest mean logp decrease/increase across samples"
        }
    }
    
    with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_dir / 'statistics.json'}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nSample Level ({stats['sample_level']['total_samples']} samples):")
    print(f"  Joint LogP Diff: {stats['sample_level']['joint_logp_diff']['mean']:.4f} ± {stats['sample_level']['joint_logp_diff']['std']:.4f}")
    print(f"  Improved: {stats['sample_level']['joint_logp_diff']['improved_samples']} ({100*stats['sample_level']['joint_logp_diff']['improved_samples']/stats['sample_level']['total_samples']:.1f}%)")
    print(f"  Degraded: {stats['sample_level']['joint_logp_diff']['degraded_samples']} ({100*stats['sample_level']['joint_logp_diff']['degraded_samples']/stats['sample_level']['total_samples']:.1f}%)")
    
    print(f"\nToken Level - All Tokens ({stats['token_level']['all_tokens']['total_tokens']} tokens):")
    print(f"  Token LogP Diff: {stats['token_level']['all_tokens']['logp_diff']['mean']:.4f} ± {stats['token_level']['all_tokens']['logp_diff']['std']:.4f}")
    print(f"  Improved: {stats['token_level']['all_tokens']['logp_diff']['improved_tokens']} ({100*stats['token_level']['all_tokens']['logp_diff']['improved_tokens']/stats['token_level']['all_tokens']['total_tokens']:.1f}%)")
    print(f"  Degraded: {stats['token_level']['all_tokens']['logp_diff']['degraded_tokens']} ({100*stats['token_level']['all_tokens']['logp_diff']['degraded_tokens']/stats['token_level']['all_tokens']['total_tokens']:.1f}%)")
    
    if filtered_token_diffs:
        print(f"\nToken Level - Filtered (|diff| >= 0.1) ({stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['total_tokens']} tokens):")
        print(f"  Token LogP Diff: {stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['mean']:.4f} ± {stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['std']:.4f}")
        print(f"  Improved: {stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['improved_tokens']} ({100*stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['improved_tokens']/stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['total_tokens']:.1f}%)")
        print(f"  Degraded: {stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['degraded_tokens']} ({100*stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['logp_diff']['degraded_tokens']/stats['token_level']['filtered_tokens_abs_diff_gte_0.1']['total_tokens']:.1f}%)")
    
    
    print(f"\nPosition Level Analysis:")
    print(f"  Positions with decrease: {len(decrease_positions)}")
    print(f"  Positions with increase: {len(increase_positions)}")
    if top_decrease_positions:
        print(f"  Top decrease position: {top_decrease_positions[0]['position']} (mean diff: {top_decrease_positions[0]['mean_diff']:.4f})")
    if top_increase_positions:
        print(f"  Top increase position: {top_increase_positions[0]['position']} (mean diff: {top_increase_positions[0]['mean_diff']:.4f})")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Path to comparison results JSON")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Directory to save visualizations")
    
    args = parser.parse_args()
    visualize_results(args.results_path, args.output_dir)
