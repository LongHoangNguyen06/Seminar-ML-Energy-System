import math
import os

import matplotlib.pyplot as plt


def plot_df(df, df_name, CONF):
    """Plot all columns of a DataFrame in separate
    subplots within a single figure and save the plot
    to a directory.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        df_name (str): Name of the DataFrame.
        CONF (config.Config): Configuration object.
    """
    num_columns = (
        len(df.columns) - 2
    )  # Exclude 'Date to' from the count if it's not plotted
    nrows = 4
    ncols = math.ceil(num_columns / nrows)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(20, 10), constrained_layout=True
    )
    fig.suptitle("Overview of All Columns")

    plot_dir = os.path.join(CONF.data.data_dir, "processed_data_figures")
    os.makedirs(plot_dir, exist_ok=True)

    axes = axes.flatten()

    # Extract 'Date from' for use as x-axis
    x_dates = df["Date from"]

    # Plot each column except 'Date from' and 'Date to'
    for idx, column in enumerate(df.columns.drop(["Date from", "Date to"])):
        ax = axes[idx]
        ax.plot(x_dates, df[column], label=column)
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()

    # Hide unused subplots
    unused = len(df.columns) - 2
    for ax in axes[unused:]:
        ax.axis("off")

    plt.savefig(os.path.join(plot_dir, f"{df_name.replace('/', '-')}.png"))
    plt.close()
