#!/usr/bin/env python3
"""
Generate single benchmark results chart for GPU-accelerated neural network project
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional look
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Your actual benchmark data from README
operations = ['Matrix Mult\n(512×512)', 'Matrix Mult\n(256×256)', 'Activation\n(1M elements)']
cpu_times = [105.49, 11.61, 2.92]  # CPU times in ms from README
gpu_times = [3.86, 1.26, 1.29]     # GPU times in ms from README
speedups = [27.34, 9.33, 2.27]     # Calculated speedups from README

# Create single figure with professional styling
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle('GPU vs CPU Performance Benchmark Results', fontsize=20, fontweight='bold', y=0.95)

# Create grouped bar chart
x = np.arange(len(operations))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU', color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add speedup line on secondary y-axis
ax2 = ax.twinx()
line = ax2.plot(x, speedups, 'o-', color='#6A4C93', linewidth=3, markersize=8, 
                label='Speedup Factor', markerfacecolor='white', markeredgecolor='#6A4C93', markeredgewidth=2)

# Customize the main chart
ax.set_xlabel('Operations', fontsize=14, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold', color='#333333')
ax.set_title('Performance Comparison with GPU Speedup', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(operations, fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(cpu_times) * 1.15)

# Customize the secondary y-axis
ax2.set_ylabel('Speedup Factor (×)', fontsize=14, fontweight='bold', color='#6A4C93')
ax2.set_ylim(0, max(speedups) * 1.15)
ax2.tick_params(axis='y', labelcolor='#6A4C93')

# Add value labels on CPU bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.2f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#FF6B6B')

# Add value labels on GPU bars
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#4ECDC4')

# Add speedup labels on the line
for i, speedup in enumerate(speedups):
    ax2.text(i, speedup + 0.5, f'{speedup:.2f}×', ha='center', va='bottom', 
             fontsize=11, fontweight='bold', color='#6A4C93')

# Add legends
ax.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)

# Add a horizontal line at 1x speedup for reference
ax2.axhline(y=1, color='#666666', linestyle='--', alpha=0.7, linewidth=1)

# Add project info as text box
project_info = """GPU-Accelerated Neural Network Primitives
OpenCL Implementation • C++17 • 27× Speedup Achieved"""
fig.text(0.5, 0.02, project_info, ha='center', fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9))

# Adjust layout
plt.tight_layout()

# Save the chart
plt.savefig('benchmark_single.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('benchmark_single.pdf', bbox_inches='tight', facecolor='white')

print("Single benchmark chart saved as 'benchmark_single.png' and 'benchmark_single.pdf'")
print("Chart shows:")
print(f"  - Matrix Multiplication (512×512): {speedups[0]:.2f}× speedup")
print(f"  - Matrix Multiplication (256×256): {speedups[1]:.2f}× speedup") 
print(f"  - Activation Functions (1M elements): {speedups[2]:.2f}× speedup")

plt.show()
