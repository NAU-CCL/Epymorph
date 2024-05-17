"""
Contexts which provide console messaging for epymorph processes like simulation runs and geo fetching.
It's nice to have some console output to show progress during long running tasks!
"""
from contextlib import contextmanager
from time import perf_counter
from typing import Generator

from epymorph.event import (AdrioStart, DynamicGeoEvents, FetchStart, OnStart,
                            OnTick, SimulationEvents)
from epymorph.util import progress, subscriptions


@contextmanager
def sim_messaging(sim: SimulationEvents, geo_messaging=False) -> Generator[None, None, None]:
    """
    Attach fancy console messaging to a Simulation such as a progress bar.
    If `geo_messaging` is true, provide verbose messaging about geo operations
    (if applicable, e.g., when fetching external data).
    """

    start_time: float | None = None

    # If geo_messaging is true, the user has requested verbose messaging re: geo operations.
    # However we don't want to make a strong assertion that a sim has a geo, nor what type that geo is.
    # So we'll do this dynamically!
    # - if we have a geo, and
    # - if it's an instance of DynamicGeoEvents, and
    # - if the user has enabled geo messaging, then and only then will we subscribe to its adrio_start event
    sim_geo = None
    if hasattr(sim, 'geo'):
        sim_geo = getattr(sim, 'geo')

    def on_start(ctx: OnStart) -> None:
        start_date = ctx.dim.start_date
        end_date = ctx.dim.end_date
        duration_days = ctx.dim.days

        print(f"Running simulation ({sim.__class__.__name__}):")
        print(f"• {start_date} to {end_date} ({duration_days} days)")
        print(f"• {ctx.dim.nodes} geo nodes")
        print(progress(0.0), end='\r')

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: OnTick) -> None:
        print(progress(tick.percent_complete), end='\r')

    def on_end(_: None) -> None:
        end_time = perf_counter()
        print(progress(1.0))
        if start_time is not None:
            print(f"Runtime: {(end_time - start_time):.3f}s")

    def adrio_start(adrio: AdrioStart) -> None:
        print(f"Uncached geo attribute requested: {adrio.attribute}. Retreiving now...")

    # Set up a subscriptions context, subscribe our handlers,
    # then yield to the outer context (ostensibly where the sim will be run).
    with subscriptions() as subs:
        subs.subscribe(sim.on_start, on_start)
        subs.subscribe(sim.on_tick, on_tick)
        subs.subscribe(sim.on_end, on_end)
        if geo_messaging and isinstance(sim_geo, DynamicGeoEvents):
            print("Geo not loaded from cache; "
                  "attributes will be lazily loaded during simulation run.")
            subs.subscribe(sim_geo.adrio_start, adrio_start)
        yield  # to outer context


@contextmanager
def dynamic_geo_messaging(dyn: DynamicGeoEvents) -> Generator[None, None, None]:
    """
    Attach progress messaging to a DynamicGeo for verbose printing of data retreival progress.
    Creates subscriptions on the Geo's events.
    """

    start_time: float | None = None

    def fetch_start(event: FetchStart) -> None:
        print("Fetching dynamic geo data")
        print(f"• {event.adrio_len} attributes")

        nonlocal start_time
        start_time = perf_counter()

    def adrio_start(event: AdrioStart) -> None:
        msg = f"Fetching {event.attribute}..."
        if event.index is not None and event.adrio_len is not None:
            msg = f"{msg} [{event.index + 1}/{event.adrio_len}]"
        print(msg)

    def fetch_end(_: None) -> None:
        print("Complete.")
        end_time = perf_counter()
        if start_time is not None:
            print(f"Total fetch time: {(end_time - start_time):.3f}s")

    with subscriptions() as subs:
        subs.subscribe(dyn.fetch_start, fetch_start)
        subs.subscribe(dyn.adrio_start, adrio_start)
        subs.subscribe(dyn.fetch_end, fetch_end)
        yield  # to outer context
