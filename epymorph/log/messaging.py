"""
Contexts which provide console messaging for epymorph processes like simulation runs and geo fetching.
It's nice to have some console output to show progress during long running tasks!
"""
from contextlib import contextmanager
from time import perf_counter
from typing import Generator

from epymorph.event import AdrioFinish, AdrioStart, EventBus, OnStart, OnTick
from epymorph.util import progress, subscriptions

_events = EventBus()


@contextmanager
def sim_messaging(adrio=True) -> Generator[None, None, None]:
    """
    Produce console messaging during simulation runs, like a progress bar.
    If `adrio` is True: display when ADRIOs are fetching data.
    """

    start_time: float | None = None

    def on_start(e: OnStart) -> None:
        start_date = e.dim.start_date
        end_date = e.dim.end_date
        duration_days = e.dim.days

        print(f"Running simulation ({e.simulator}):")
        print(f"• {start_date} to {end_date} ({duration_days} days)")
        print(f"• {e.dim.nodes} geo nodes")
        print(progress(0.0), end="\r")

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: OnTick) -> None:
        print(progress(tick.percent_complete), end="\r")

    def on_finish(_: None) -> None:
        end_time = perf_counter()
        print(progress(1.0))
        if start_time is not None:
            print(f"Runtime: {(end_time - start_time):.3f}s")

    def on_adrio_start(e: AdrioStart) -> None:
        print(f"ADRIO {e.adrio_name} fetching `{e.attribute}`...", end="")

    def on_adrio_finish(e: AdrioFinish) -> None:
        print(f" done ({e.duration:.3f} seconds)")

    with subscriptions() as subs:
        # Set up a subscriptions context, subscribe our handlers,
        # then yield to the outer context (ostensibly where the sim will be run).
        subs.subscribe(_events.on_start, on_start)
        subs.subscribe(_events.on_tick, on_tick)
        subs.subscribe(_events.on_finish, on_finish)
        if adrio:
            subs.subscribe(_events.on_adrio_start, on_adrio_start)
            subs.subscribe(_events.on_adrio_finish, on_adrio_finish)
        yield  # to outer context
        # And now our event handlers will be unsubscribed.
