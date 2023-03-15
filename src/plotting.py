import matplotlib.pyplot as plt

import simulation as S
from util import stridesum


def plot_pop_prevalence(out: S.Output, pop_idx: int, labels: dict[str, str], scaling) -> None:
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
    if 'title' in labels:
        ax.set_title(labels['title'])
    if 'y_label' in labels:
        ax.set_ylabel(labels['y_label'])
    if 'x_label' in labels:
        ax.set_xlabel(labels['x_label'])
    plt.show()


def plot_events(out: S.Output, event_idx: int, labels: dict[str, str], scaling) -> None:
    # When plotting events, aggregate to the day. Uneven taus produce sawtooth graphs.
    day_axis = list(range(out.clock.num_days))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for pop_idx, pop_inc in enumerate(out.incidence):
        values = stridesum(pop_inc[:, event_idx], out.clock.num_steps)
        y_axis = scaling(pop_idx, values)
        ax.plot(day_axis, y_axis, label=out.pop_labels[pop_idx])
    ax.legend()
    if 'title' in labels:
        ax.set_title(labels['title'])
    if 'y_label' in labels:
        ax.set_ylabel(labels['y_label'])
    if 'x_label' in labels:
        ax.set_xlabel(labels['x_label'])
    plt.show()
