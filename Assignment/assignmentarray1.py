import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-5, 5, 3)
lines = [
    {"m": 2, "c": 1, "color": "black", "linestyle": "-.", "label": "y = 2x + 1"},
    {"m": 2, "c": 2, "color": "blue", "linestyle": "--", "label": "y = 2x + 2"},
    {"m": 2, "c": 3, "color": "green", "linestyle": ":", "label": "y = 2x + 3"},
]

plt.figure(figsize=(7, 5))
for line in lines:
    y = line["m"] * x + line["c"]
    plt.plot(x, y, color=line["color"], linestyle=line["linestyle"], label=line["label"])

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Lines: y = 2x + c')


plt.legend()
plt.grid(True)
plt.show()
