# import matplotlib.pyplot as plt
# import numpy as np

# # Data extracted from the table provided by the user
# t_values = [5, 10, 15]
# categories = ['A', 'B', 'C', 'A + B + C']

# # Precision, Recall, and F1-scores for each category and threshold
# # Data structured as [precision, recall, f1-score] for each category at each threshold
# data = {
#     5: {
#         'A': [100, 71.43, 83.33],
#         'B': [40, 28.57, 33.33],
#         'C': [20, 14.29, 16.67],
#         'A + B + C': [80, 18.18, 29.63]
#     },
#     10: {
#         'A': [100, 71.43, 83.33],
#         'B': [40, 28.57, 33.33],
#         'C': [0, 0, 0],
#         'A + B + C': [80, 18.18, 29.63]
#     },
#     15: {
#         'A': [100, 71.43, 83.33],
#         'B': [40, 28.57, 33.33],
#         'C': [0, 0, 0],
#         'A + B + C': [100, 22.73, 37.04]
#     }
# }

# fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# for idx, t in enumerate(t_values):
#     ax = axes[idx]
#     precision = [data[t][cat][0] for cat in categories]
#     recall = [data[t][cat][1] for cat in categories]
#     f1_score = [data[t][cat][2] for cat in categories]
    
#     bar_width = 0.25
#     index = np.arange(len(categories))
    
#     rects1 = ax.bar(index, precision, bar_width, label='Precision')
#     rects2 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
#     rects3 = ax.bar(index + 2*bar_width, f1_score, bar_width, label='F1-score')

#     ax.set_xlabel('Categories')
#     ax.set_title(f'Performance Metrics at t={t}')
#     ax.set_xticks(index + bar_width)
#     ax.set_xticklabels(categories)
#     ax.legend()

# plt.tight_layout()
# plt.show()
