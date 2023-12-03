import numpy as np
import matplotlib.pyplot as plt

def draw_losses(losses, house):

    total = len(losses)

    plt.plot(np.linspace(0, total, total), losses, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('step')
    plt.ylabel('Losses')
    plt.legend()
    # plt.show()
    plt.savefig(f'{house}.png')
