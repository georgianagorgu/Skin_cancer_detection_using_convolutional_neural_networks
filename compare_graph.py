import matplotlib.pyplot as plt
import numpy as np

# valorile tale (NU le schimba)
models = ['EfficientNetB0', 'ResNet50']
accuracy = [0.8545, 0.8742]
precision = [0.8187, 0.8974]
recall = [0.8733, 0.8167]
f1_score = [0.8452, 0.8551]

x = np.arange(len(models))
width = 0.2

plt.figure()

plt.bar(x - width, accuracy, width, label='Accuracy')
plt.bar(x, precision, width, label='Precision')
plt.bar(x + width, recall, width, label='Recall')
plt.bar(x + 2*width, f1_score, width, label='F1 Score')

plt.xticks(x, models)
plt.title("Comparatie performanta modele")
plt.ylabel("Valori")
plt.legend()

# salvare (IMPORTANT)
plt.savefig("results/plots/model_comparison.png")

plt.show()