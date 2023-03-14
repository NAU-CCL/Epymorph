from math import log
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import simulation as S
from util import stridesum


def plot_pop_prevalence(out: S.Output, pop_idx: int, scaling) -> None:
    # TODO: assumes three events
    time_axis = [t.tausum for t in out.clock.ticks]
    s = [scaling(x) for x in out.prevalence[pop_idx][:, 0]]
    i = [scaling(x) for x in out.prevalence[pop_idx][:, 1]]
    r = [scaling(x) for x in out.prevalence[pop_idx][:, 2]]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_axis, s, color='tab:blue')
    ax.plot(time_axis, i, color='tab:red')
    ax.plot(time_axis, r, color='tab:orange')
    plt.show()


def plot_events(out: S.Output, event_idx: int, scaling) -> None:
    day_axis = [t.day for t in out.clock.ticks[::2]]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for pop_idx, pop_inc in enumerate(out.incidence):
        values = stridesum(pop_inc[:, event_idx], 2)
        y_axis = scaling(pop_idx, values)
        # weight = total_pop[pop_idx] / 100_000
        ax.plot(day_axis, y_axis, label=out.pop_labels[pop_idx])
    ax.legend()
    plt.show()
