"""
TOPSIS Analysis for Best Pre-trained Models for Text Classification
Roll Number: 442 (ending with 2 - Text Classification task)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from topsis import TOPSIS

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


def create_model_data():
    """
    Create decision matrix for pre-trained text classification models.
    
    Criteria:
    1. Accuracy (max) - Classification accuracy on standard benchmarks
    2. Inference Speed (max) - Tokens per second (higher is better)
    3. Model Size (min) - Size in MB (lower is better)
    4. Memory Usage (min) - Peak memory in GB (lower is better)
    5. Training Time (min) - Hours to fine-tune (lower is better)
    6. F1 Score (max) - F1 score on test set (higher is better)
    """
    
    models = [
        "BERT-base",
        "BERT-large",
        "RoBERTa-base",
        "RoBERTa-large",
        "DistilBERT",
        "ALBERT-base",
        "ALBERT-xxlarge",
        "ELECTRA-base",
        "DeBERTa-base",
        "DeBERTa-large"
    ]
    
    # Decision Matrix
    # [Accuracy (%), Speed (tokens/sec), Size (MB), Memory (GB), Training Time (hrs), F1 Score]
    decision_matrix = np.array([
        [92.5, 450, 440, 2.1, 8.5, 0.925],      # BERT-base
        [94.2, 280, 1300, 4.8, 15.2, 0.942],   # BERT-large
        [93.8, 420, 500, 2.3, 9.1, 0.938],     # RoBERTa-base
        [95.1, 260, 1500, 5.2, 16.8, 0.951],   # RoBERTa-large
        [91.2, 680, 260, 1.2, 4.2, 0.912],     # DistilBERT
        [92.8, 480, 230, 1.8, 6.8, 0.928],     # ALBERT-base
        [95.5, 200, 2200, 7.5, 22.5, 0.955],   # ALBERT-xxlarge
        [93.5, 520, 420, 2.0, 7.5, 0.935],     # ELECTRA-base
        [94.5, 380, 580, 2.5, 10.5, 0.945],    # DeBERTa-base
        [95.8, 240, 1400, 5.5, 18.2, 0.958]    # DeBERTa-large
    ])
    
    criteria = [
        "Accuracy (%)",
        "Inference Speed (tokens/sec)",
        "Model Size (MB)",
        "Memory Usage (GB)",
        "Training Time (hours)",
        "F1 Score"
    ]
    
    criteria_types = ['max', 'max', 'min', 'min', 'min', 'max']
    
    # Weights for criteria (sum should be 1.0)
    # Higher weight on accuracy and F1 score, moderate on speed, lower on resource usage
    weights = np.array([0.25, 0.15, 0.10, 0.10, 0.15, 0.25])
    
    return models, decision_matrix, criteria, criteria_types, weights


def run_topsis_analysis():
    """Run TOPSIS analysis and return results."""
    models, decision_matrix, criteria, criteria_types, weights = create_model_data()
    
    # Create TOPSIS instance
    topsis = TOPSIS(decision_matrix, weights, criteria_types)
    
    # Perform ranking
    results = topsis.rank()
    
    return models, decision_matrix, criteria, criteria_types, weights, results


def create_results_dataframe(models, decision_matrix, criteria, results):
    """Create a comprehensive results DataFrame."""
    df_results = pd.DataFrame(decision_matrix, index=models, columns=criteria)
    df_results['TOPSIS Score'] = results['scores']
    df_results['Rank'] = [list(results['rankings']).index(i) + 1 for i in range(len(models))]
    df_results = df_results.sort_values('Rank')
    
    return df_results


def visualize_results(models, decision_matrix, criteria, results):
    """Create comprehensive visualizations."""
    
    # Create results DataFrame
    df_results = create_results_dataframe(models, decision_matrix, criteria, results)
    
    # 1. TOPSIS Scores Bar Chart
    plt.figure(figsize=(12, 6))
    sorted_indices = np.argsort(results['scores'])[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = results['scores'][sorted_indices]
    
    bars = plt.barh(range(len(sorted_models)), sorted_scores, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_models))))
    plt.yticks(range(len(sorted_models)), sorted_models)
    plt.xlabel('TOPSIS Score', fontsize=12, fontweight='bold')
    plt.title('TOPSIS Scores for Text Classification Models\n(Higher is Better)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        plt.text(score + 0.01, i, f'{score:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('topsis_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Criteria Comparison Radar Chart (Top 5 models)
    top_5_indices = results['rankings'][:5]
    top_5_models = [models[i] for i in top_5_indices]
    
    # Normalize criteria for radar chart (0-1 scale)
    normalized_data = []
    for idx in top_5_indices:
        model_data = decision_matrix[idx]
        normalized = []
        for i, criterion_type in enumerate(['max', 'max', 'min', 'min', 'min', 'max']):
            if criterion_type == 'max':
                normalized.append(model_data[i] / decision_matrix[:, i].max())
            else:
                normalized.append(1 - (model_data[i] - decision_matrix[:, i].min()) / 
                                (decision_matrix[:, i].max() - decision_matrix[:, i].min()))
        normalized_data.append(normalized)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_5_models)))
    for i, (model, data, color) in enumerate(zip(top_5_models, normalized_data, colors)):
        data += data[:1]  # Complete the circle
        ax.plot(angles, data, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, data, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Top 5 Models - Criteria Comparison (Normalized)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('radar_chart_top5.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance vs Efficiency Scatter Plot
    plt.figure(figsize=(12, 8))
    
    # X-axis: Average of normalized accuracy and F1 (performance)
    # Y-axis: Average of normalized speed and inverse of size/memory (efficiency)
    performance = (decision_matrix[:, 0] / 100 + decision_matrix[:, 5]) / 2
    efficiency = (decision_matrix[:, 1] / decision_matrix[:, 1].max() + 
                  (1 - decision_matrix[:, 2] / decision_matrix[:, 2].max()) +
                  (1 - decision_matrix[:, 3] / decision_matrix[:, 3].max())) / 3
    
    scatter = plt.scatter(performance, efficiency, 
                         s=results['scores']*1000, 
                         c=results['scores'], 
                         cmap='viridis',
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=2)
    
    # Add labels
    for i, model in enumerate(models):
        plt.annotate(model, (performance[i], efficiency[i]), 
                    fontsize=9, ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Performance (Avg of Accuracy & F1 Score)', fontsize=12, fontweight='bold')
    plt.ylabel('Efficiency (Speed, Size, Memory)', fontsize=12, fontweight='bold')
    plt.title('Performance vs Efficiency Analysis\n(Bubble size = TOPSIS Score)', 
              fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='TOPSIS Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Criteria Weights Visualization
    plt.figure(figsize=(10, 6))
    weights = np.array([0.25, 0.15, 0.10, 0.10, 0.15, 0.25])
    bars = plt.barh(criteria, weights, color=plt.cm.coolwarm(np.linspace(0, 1, len(criteria))))
    plt.xlabel('Weight', fontsize=12, fontweight='bold')
    plt.title('Criteria Weights in TOPSIS Analysis', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, weight in zip(bars, weights):
        plt.text(weight + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{weight:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('criteria_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Distance from Ideal Solutions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sorted_indices = np.argsort(results['scores'])[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    pos_distances = results['positive_distances'][sorted_indices]
    neg_distances = results['negative_distances'][sorted_indices]
    
    x = np.arange(len(sorted_models))
    width = 0.35
    
    ax1.barh(x - width/2, pos_distances, width, label='Distance from Positive Ideal', 
             color='coral', alpha=0.8)
    ax1.barh(x + width/2, neg_distances, width, label='Distance from Negative Ideal', 
             color='lightblue', alpha=0.8)
    ax1.set_yticks(x)
    ax1.set_yticklabels(sorted_models)
    ax1.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Distance from Ideal Solutions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Ratio plot
    ratio = neg_distances / (pos_distances + neg_distances)
    ax2.barh(range(len(sorted_models)), ratio, color=plt.cm.viridis(ratio))
    ax2.set_yticks(range(len(sorted_models)))
    ax2.set_yticklabels(sorted_models)
    ax2.set_xlabel('Closeness Coefficient (TOPSIS Score)', fontsize=12, fontweight='bold')
    ax2.set_title('Closeness to Positive Ideal Solution', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, r in enumerate(ratio):
        ax2.text(r + 0.01, i, f'{r:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ideal_distances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved successfully!")


def generate_results_table(models, decision_matrix, criteria, results):
    """Generate and save results table."""
    df_results = create_results_dataframe(models, decision_matrix, criteria, results)
    
    # Save to CSV
    df_results.to_csv('topsis_results.csv')
    
    # Create formatted table for display
    print("\n" + "="*100)
    print("TOPSIS ANALYSIS RESULTS - TEXT CLASSIFICATION MODELS")
    print("="*100)
    print("\nFinal Rankings:\n")
    print(df_results.to_string())
    
    print("\n" + "="*100)
    print("TOP 3 RECOMMENDED MODELS:")
    print("="*100)
    for i, rank in enumerate(results['rankings'][:3], 1):
        model_idx = rank
        print(f"\n{i}. {models[model_idx]}")
        print(f"   TOPSIS Score: {results['scores'][model_idx]:.4f}")
        print(f"   Accuracy: {decision_matrix[model_idx, 0]:.1f}%")
        print(f"   F1 Score: {decision_matrix[model_idx, 5]:.3f}")
        print(f"   Inference Speed: {decision_matrix[model_idx, 1]:.0f} tokens/sec")
        print(f"   Model Size: {decision_matrix[model_idx, 2]:.0f} MB")
    
    return df_results


def main():
    """Main function to run the TOPSIS analysis."""
    print("Starting TOPSIS Analysis for Text Classification Models...")
    print("Roll Number: 442 (Text Classification Task)\n")
    
    # Run TOPSIS analysis
    models, decision_matrix, criteria, criteria_types, weights, results = run_topsis_analysis()
    
    # Generate results table
    df_results = generate_results_table(models, decision_matrix, criteria, results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(models, decision_matrix, criteria, results)
    
    print("\n" + "="*100)
    print("Analysis Complete! Check the generated files:")
    print("  - topsis_results.csv: Detailed results table")
    print("  - topsis_scores.png: TOPSIS scores visualization")
    print("  - radar_chart_top5.png: Top 5 models comparison")
    print("  - performance_efficiency.png: Performance vs efficiency analysis")
    print("  - criteria_weights.png: Criteria weights visualization")
    print("  - ideal_distances.png: Distance from ideal solutions")
    print("="*100)


if __name__ == "__main__":
    main()
