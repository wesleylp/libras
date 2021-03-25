import seaborn as sns


def set_plot_config():
    sns.set_context('paper',
                    font_scale=2,
                    rc={
                        'lines.linewidth': 2,
                        'text.usetex': True,
                        'image.interpolation': 'nearest',
                        'image.cmap': 'gray',
                        'figure.figsize': (10.0, 8.0)
                    })
    sns.set_style(style='ticks')
    sns.set_palette('colorblind', color_codes=True)
