import matplotlib.pyplot as plt

def plot_lorenz(data, save_path):
    f = plt.figure(figsize=(6, 4))
    plt.plot(data[:, 0], data[:, 2], label='Ground truth')
    plt.plot(data[0, 0], data[0, 2], "ko", label="Initial condition", markersize=8)
    plt.legend()
    #plt.show()
    print(f'Saving plot {save_path}')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(f)