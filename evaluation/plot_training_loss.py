import matplotlib.pyplot as plt

# Manually enter your epoch losses
losses = [0.7961, 0.68, 0.61, 0.55, 0.5006]

epochs = range(1, len(losses) + 1)

plt.figure()
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.show()
