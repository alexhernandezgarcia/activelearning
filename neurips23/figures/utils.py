import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import hsv_to_rgb
from omegaconf.listconfig import ListConfig


def plot_setup():
    sns.reset_orig()
#     sns.set(style="whitegrid")
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update(
        {
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "Utopia",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "ITC Bookman",
                "Bookman",
                "Times",
                "Palatino",
                "Charter",
                "serif" "Bitstream Vera Serif",
                "DejaVu Serif",
            ]
        }
    )


def get_hue_palette(palette, n_hues):
    if isinstance(palette, list) or isinstance(palette, ListConfig):
        palettes = [
            sns.color_palette(p, as_cmap=False, n_colors=n_hue)
            for p, n_hue in zip(palette, n_hues)
        ]
        palettes = [p for pl in palettes for p in pl]
        return palettes
    else:
        return sns.color_palette(palette, as_cmap=False, n_colors=n_hue)

def get_pkl(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                return os.path.join(root, file)
    return None
