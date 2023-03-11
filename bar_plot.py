import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
DEV = [0.731]
TEST = [0.952]
 
# Set position of bar on X axis
br1 = np.arange(len(DEV))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, DEV, color ='r', width = barWidth,
        edgecolor ='grey', label ='IT')
plt.bar(br2, TEST, color ='g', width = barWidth,
        edgecolor ='grey', label ='ECE')
 
# Adding Xticks
plt.xlabel('Model', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(DEV))],
        ['distilBERT-base-uncased'])
 
plt.legend()
plt.savefig('figures/three_eight.png')
