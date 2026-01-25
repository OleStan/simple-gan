#!/usr/bin/env python
"""Compare different architectural approaches for dual-profile generation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_approaches():
    """Visualize the three architectural approaches considered."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    approaches = [
        {
            'title': 'Approach 1: Two Independent GANs',
            'description': [
                'Generator₁ → σ profiles',
                'Generator₂ → μ profiles',
                '',
                'Pros:',
                '  • Simple architecture',
                '  • Independent training',
                '',
                'Cons:',
                '  • No correlation capture',
                '  • 2× parameters',
                '  • Physically inconsistent'
            ]
        },
        {
            'title': 'Approach 2: Single Generator',
            'description': [
                'Generator → [σ, μ] concatenated',
                '',
                'Pros:',
                '  • Simple architecture',
                '  • Captures correlations',
                '',
                'Cons:',
                '  • No specialization',
                '  • Multi-scale challenges',
                '  • Single optimization path'
            ]
        },
        {
            'title': 'Approach 3: Dual-Head (SELECTED)',
            'description': [
                'Shared Encoder → Features',
                '  ├─ σ Head → σ profiles',
                '  └─ μ Head → μ profiles',
                '',
                'Pros:',
                '  • Captures correlations',
                '  • Specialized heads',
                '  • Multi-scale handling',
                '  • Interpretable',
                '',
                'Cons:',
                '  • Slightly more complex'
            ]
        }
    ]
    
    for i, (ax, approach) in enumerate(zip(axes, approaches)):
        ax.text(0.5, 0.95, approach['title'], 
               ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        
        y_pos = 0.85
        for line in approach['description']:
            ax.text(0.05, y_pos, line, 
                   ha='left', va='top', fontsize=9,
                   transform=ax.transAxes,
                   family='monospace')
            y_pos -= 0.05
        
        if i == 2:
            ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                      fill=False, edgecolor='green',
                                      linewidth=3, transform=ax.transAxes))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    
    output_dir = Path('./results/comparison_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'architecture_comparison.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved architecture comparison to {output_dir / 'architecture_comparison.png'}")
    
    plt.close()


def create_decision_matrix():
    """Create a decision matrix comparing the approaches."""
    
    criteria = [
        'Correlation Capture',
        'Parameter Efficiency',
        'Multi-scale Handling',
        'Training Stability',
        'Interpretability',
        'Implementation Complexity'
    ]
    
    scores = {
        'Two Independent GANs': [1, 1, 2, 3, 2, 3],
        'Single Generator': [3, 3, 2, 2, 2, 3],
        'Dual-Head (Selected)': [3, 3, 3, 3, 3, 2]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(criteria))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (approach, color) in enumerate(zip(scores.keys(), colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, scores[approach], width, 
                     label=approach, color=color, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Criteria', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score (1=Poor, 3=Excellent)', fontsize=11, fontweight='bold')
    ax.set_title('Architecture Comparison Matrix', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, rotation=45, ha='right')
    ax.set_ylim(0, 3.5)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path('./results/comparison_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'decision_matrix.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved decision matrix to {output_dir / 'decision_matrix.png'}")
    
    plt.close()


def main():
    print("="*60)
    print("Generating Architecture Comparison Analysis")
    print("="*60)
    
    print("\nCreating visualizations...")
    visualize_approaches()
    create_decision_matrix()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nConclusion:")
    print("  Selected: Dual-Head Generator with Shared Encoder")
    print("  Rationale:")
    print("    • Best correlation capture via shared representation")
    print("    • Specialized heads for σ and μ optimize separately")
    print("    • Handles multi-scale outputs (10⁶ vs 10¹)")
    print("    • More interpretable than single-head approach")
    print("    • More efficient than two independent GANs")


if __name__ == '__main__':
    main()
