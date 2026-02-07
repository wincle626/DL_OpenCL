#!/usr/bin/env python3
"""
Generate benchmark results chart for GPU-accelerated neural network project
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional look
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Your actual benchmark data
operations = ['Matrix Mult\n(512×512)', 'Matrix Mult\n(256×256)', 'Activation\n(1M elements)']
cpu_times = [109.61, 11.82, 2.92]  # CPU times in ms
gpu_times = [4.93, 1.47, 1.29]     # GPU times in ms
speedups = [22.23, 8.05, 2.27]     # Calculated speedups

# Create figure with professional styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GPU vs CPU Performance Benchmark Results', fontsize=18, fontweight='bold', y=0.95)

# Left plot: Bar chart comparison
x = np.arange(len(operations))
width = 0.35

bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#4ECDC4', alpha=0.8)

# Customize the first plot
ax1.set_xlabel('Operations', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(operations, fontsize=10)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right plot: Speedup visualization
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars3 = ax2.bar(operations, speedups, color=colors, alpha=0.8)

# Customize the second plot
ax2.set_xlabel('Operations', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup Factor (×)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Speedup Over CPU', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add speedup labels on bars
for bar, speedup in zip(bars3, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{speedup:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add a horizontal line at 1x speedup for reference
ax2.axhline(y=1, color='#666666', linestyle='--', alpha=0.7, label='No Speedup (1×)')
ax2.legend(fontsize=11)

# Adjust layout and add project info
plt.tight_layout()

# Add project info as text box
project_info = """GPU-Accelerated Neural Network Primitives
OpenCL Implementation • C++17 • 22× Speedup Achieved"""
fig.text(0.5, 0.02, project_info, ha='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

# Save the chart
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('benchmark_results.pdf', bbox_inches='tight', facecolor='white')

print("Benchmark chart saved as 'benchmark_results.png' and 'benchmark_results.pdf'")
print("Chart shows:")
print(f"  - Matrix Multiplication (512×512): {speedups[0]:.2f}× speedup")
print(f"  - Matrix Multiplication (256×256): {speedups[1]:.2f}× speedup") 
print(f"  - Activation Functions (1M elements): {speedups[2]:.2f}× speedup")

plt.show()
