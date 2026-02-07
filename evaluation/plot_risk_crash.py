import numpy as np
import matplotlib.pyplot as plt

FPS = 10
THRESHOLD = 0.7

risk = np.load("risk_crash.npy")
time = [i / FPS for i in range(len(risk))]

plt.figure(figsize=(7,4))
plt.plot(time, risk, label="Predicted Risk")
plt.axhline(y=THRESHOLD, linestyle="--", label="Alert Threshold")
plt.xlabel("Time (seconds)")
plt.ylabel("Risk Score")
plt.title("Risk Score vs Time (Crash Video)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
