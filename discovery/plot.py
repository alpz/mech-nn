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


def plot_logistic(params, act_rs, act_rks, save_path):
    """ params[...,2]  r, r/k"""
    f, ax = plt.subplots(figsize=(6, 4))
    ax.plot(act_rs, label='GT R')
    ax.plot(params[:,0], label='Learned R', alpha=0.9)

    ax.plot(act_rks, label='GT -R/K')
    ax.plot(params[:,1], label='Learned -R/K', alpha=0.9)

    ax.set_ylim(-10,10)

    plt.legend()
    #plt.show()
    print(f'Saving plot {save_path}')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(f)