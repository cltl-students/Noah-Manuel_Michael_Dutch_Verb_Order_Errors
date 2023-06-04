# Noah-Manuel Michael
# Created: 03.06.2023
# Last updated: 03.06.2023
# Visualize the sentence length distribution
# This script was pair-programmed with ChatGPT

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Data
sentenceLength = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75, 81, 82, 83, 85, 95,
                  96, 100, 133, 144]
sentenceCount = [2161, 1205, 974, 1589, 2002, 2523, 2696, 3021, 3141, 3262, 3014, 2771, 2582, 2360, 2120, 1976, 1787,
                 1501, 1356, 1163, 1016, 906, 776, 654, 531, 526, 389, 360, 336, 260, 181, 198, 166, 155, 107, 89, 85,
                 62, 81, 48, 56, 45, 41, 26, 39, 26, 22, 13, 10, 16, 9, 7, 6, 8, 8, 8, 5, 7, 8, 4, 5, 6, 1, 2, 2, 4, 4,
                 2, 2, 1, 3, 1, 3, 1, 1, 4, 1, 1, 2, 1, 1, 1]

# Scatter Plot
plt.figure(figsize=(8, 6), facecolor='w')  # Set the figure size and background color
plt.scatter(sentenceLength, sentenceCount, color='darkblue', edgecolors='none', alpha=0.75)
plt.title('Sentence Length Distribution', fontsize=20, fontweight='normal')
plt.xlabel('Sentence Length', fontsize=16, fontweight='normal')
plt.ylabel('Amount of Sentences', fontsize=16, fontweight='normal')

# Formatting the axes
plt.xticks(fontsize=14)  # Change the font size of the tick labels on x-axis
plt.yticks(fontsize=14)  # Change the font size of the tick labels on y-axis
plt.box(False)  # Turn off the box around the plot

# Adding a line
plt.axvline(x=50, color='red', linewidth=2, linestyle='--')

# Set the x-axis major ticks to multiples of 10
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10))

# Add grid
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()  # Adjust the spacing between subplots

plt.savefig('sentence_length_distrib_plot.png', dpi=300, bbox_inches='tight')
