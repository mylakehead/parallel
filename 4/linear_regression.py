import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


n = np.array([1, 16, 256, 1024, 4096, 40960, 409600, 1048576, 10485760, 104857600, 1073741824])
T = np.array([2.41995e-06, 2.40803e-06, 2.15769e-06, 3.23057e-06, 3.70741e-06, 2.22087e-05,
     0.000114954, 0.000264418, 0.00250283, 0.0238965, 0.229267])

n = n.reshape(-1, 1)
reg = LinearRegression().fit(n, T)

lambda_ = reg.intercept_
beta = 1 / reg.coef_[0]

print(f"Estimated Latency (λ): {lambda_ * 1e6:.2f} μs")
print(f"Estimated Bandwidth (β): {beta / 1e6:.2f} MB/s")

plt.scatter(n, T, label="Measured Data")
plt.plot(n, reg.predict(n), color='red', label="Fitted Line")
plt.xlabel("Message Size (bytes)")
plt.ylabel("Transmission Time (s)")
plt.legend()
plt.show()
