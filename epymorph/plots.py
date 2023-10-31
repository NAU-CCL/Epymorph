"""Out built-in plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from epymorph.engine.standard_sim import Output
from epymorph.simulation import SimDType


def per_day_incidence(out: Output) -> NDArray[SimDType]:
    """
    Convert incidence from a per-tick value to a per-day value for the given event.
    Returns a shape (D,N,E) array, where D is the number of simulation days.
    """
    T, N, _, E = out.dim.TNCE
    taus = out.dim.tau_steps
    return np.sum(
        out.incidence.reshape((T // taus, taus, N, E)),
        axis=1,
        dtype=SimDType
    )


def ticks_to_days(out: Output) -> NDArray[np.float64]:
    """
    Create a series with as many values as there are simulation ticks,
    but in the scale of fractional days. That is: the cumulative sum of
    the simulation's tau step lengths across the simulation duration.
    Returns a shape (T,) array, where T is the number of simulation ticks.
    """
    return np.cumsum(np.tile(out.dim.tau_step_lengths, out.dim.days), dtype=np.float64)


def plot_event(out: Output, event_idx: int) -> None:
    """Plot the event with the given index for all locations."""
    fig, ax = plt.subplots()
    ax.set_title(f"Event occurrences by location: {out.event_labels[event_idx]}")
    ax.set_xlabel('days')
    ax.set_ylabel('occurrences')

    x_axis = np.arange(out.dim.days)
    y_axis = per_day_incidence(out)[:, :, event_idx]
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

    x_axis = ticks_to_days(out)
    y_axis = out.prevalence[:, pop_idx, :]
    ax.plot(x_axis, y_axis, label=out.compartment_labels)

    if out.dim.compartments <= 12:
        ax.legend()

    fig.tight_layout()
    plt.show()
