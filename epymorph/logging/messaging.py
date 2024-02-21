from contextlib import contextmanager
from time import perf_counter
from typing import Generator

from epymorph.simulation import (AdrioStart, DynamicGeoEvents, FetchStart,
                                 OnStart, SimTick, SimulationEvents)
from epymorph.util import progress, subscriptions


@contextmanager
def sim_messaging(sim: SimulationEvents, geo_messaging=False) -> Generator[None, None, None]:
    """
    Attach fancy console messaging to a Simulation, e.g., a progress bar.
    This creates subscriptions on `sim`'s events, so you only need to do it once
    per sim. Returns `sim` as a convenience.
    """

    start_time: float | None = None
    use_progress_bar = sim.on_tick is not None

    if geo_messaging:
        print("Geo not found in cache; geo attributes will be lazily loaded during simulation run.")

    def on_start(ctx: OnStart) -> None:
        start_date = ctx.time_frame.start_date
        duration_days = ctx.time_frame.duration_days
        end_date = ctx.time_frame.end_date

        print(f"Running simulation ({sim.__class__.__name__}):")
        print(f"• {start_date} to {end_date} ({duration_days} days)")
        print(f"• {ctx.dim.nodes} geo nodes")
        if use_progress_bar:
            print(progress(0.0), end='\r')
        else:
            print('Running...')

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: SimTick) -> None:
        print(progress(tick.percent_complete), end='\r')

    def adrio_start(adrio: AdrioStart) -> None:
        print(f"Uncached geo attribute requested: {adrio.attribute}. Retreiving now...")

    def on_end(_: None) -> None:
        end_time = perf_counter()
        if use_progress_bar:
            print(progress(1.0))
        else:
            print('Complete.')
        if start_time is not None:
            print(f"Runtime: {(end_time - start_time):.3f}s")

    # Set up a subscriptions context, subscribe our handlers,
    # then yield to the outer context (ostensibly where the sim will be run).
    with subscriptions() as subs:
        subs.subscribe(sim.on_start, on_start)
        if sim.on_tick is not None:
            subs.subscribe(sim.on_tick, on_tick)
        if geo_messaging:
            subs.subscribe(sim.adrio_start, adrio_start)
        subs.subscribe(sim.on_end, on_end)
        yield


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
        print(f"Fetching {event.attribute}...[{event.index}/{event.adrio_len}]")

    def fetch_end(_: None) -> None:
        print("Complete.")
        end_time = perf_counter()
        if start_time is not None:
            print(f"Total fetch time: {(end_time - start_time):.3f}s")

    with subscriptions() as subs:
        subs.subscribe(dyn.fetch_start, fetch_start)
        subs.subscribe(dyn.adrio_start, adrio_start)
        subs.subscribe(dyn.fetch_end, fetch_end)
        yield
