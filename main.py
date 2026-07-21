import numpy as np
import matplotlib.pyplot as plt

arr = np.load(r".\data\unitcell\threshold\segmentation_threshold_0p005891649.npy")

print(arr.shape, arr.dtype)

if arr.ndim == 1:
    plt.plot(arr)

elif arr.ndim == 2:
    plt.imshow(arr, cmap="gray")
    plt.colorbar()

elif arr.ndim == 3:
    if arr.shape[-1] in [3, 4]:
        plt.imshow(arr)
    elif arr.shape[0] in [3, 4]:
        plt.imshow(np.transpose(arr, (1, 2, 0)))
    else:
        plt.imshow(arr[arr.shape[0] // 2], cmap="gray")

else:
    print("Unsupported shape:", arr.shape)

plt.show()