"""Our built-in plotting functions and helpful utilities."""
import matplotlib.pyplot as plt
import numpy as np

from epymorph.engine.standard_sim import Output


def plot_event(out: Output, event_idx: int) -> None:
    """Plot the event with the given index for all locations."""
    fig, ax = plt.subplots()
    ax.set_title(f"Event occurrences by location: {out.event_labels[event_idx]}")
    ax.set_xlabel('days')
    ax.set_ylabel('occurrences')

    x_axis = np.arange(out.dim.days)
    y_axis = out.incidence_per_day[:, :, event_idx]
    ax.plot(x_axis, y_axis, label=out.geo_labels)

    if out.dim.nodes <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_pop(out: Output, pop_idx: int, log_scale: bool = False) -> None:
    """Plot all compartments for the population at the given index."""
    fig, ax = plt.subplots()
    ax.set_title(f"Prevalence by compartment: {out.geo_labels[pop_idx]}")
    ax.set_xlabel('days')
    if not log_scale:
        ax.set_ylabel('individuals')
    else:
        ax.set_ylabel('log(individuals)')
        ax.set_yscale('log')

    x_axis = out.ticks_in_days
    y_axis = out.prevalence[:, pop_idx, :]
    ax.plot(x_axis, y_axis, label=out.compartment_labels)

    if out.dim.compartments <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()
