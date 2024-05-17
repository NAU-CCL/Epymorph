"""For logging epymorph simulations to a file."""
from contextlib import contextmanager
from logging import (BASIC_FORMAT, DEBUG, NOTSET, FileHandler, Formatter,
                     getLogger)
from time import perf_counter
from typing import Generator

from epymorph.event import (AdrioStart, DynamicGeoEvents, OnMovementClause,
                            OnMovementFinish, OnMovementStart, OnStart, OnTick,
                            SimWithEvents)
from epymorph.util import subscriptions


@contextmanager
def file_log(
    sim: SimWithEvents,
    log_file: str = 'debug.log',
    log_level: str | int = DEBUG,
) -> Generator[None, None, None]:
    """Attach file logging to a simulation."""

    # Initialize the logging system and create some Loggers for epymorph subsystems.
    log_handler = FileHandler(log_file, "w", "utf8")
    log_handler.setFormatter(Formatter(BASIC_FORMAT))

    epy_log = getLogger('epymorph')
    epy_log.addHandler(log_handler)
    epy_log.setLevel(log_level)

    sim_log = epy_log.getChild('sim')
    geo_log = epy_log.getChild('geo')
    mm_log = epy_log.getChild('movement')

    # Define handlers for each of the events we're interested in.

    start_time: float | None = None

    def on_start(ctx: OnStart) -> None:
        start_date = ctx.dim.start_date
        end_date = ctx.dim.end_date
        duration_days = ctx.dim.days

        sim_log.info(f"Running simulation ({sim.__class__.__name__}):")
        sim_log.info(f"- {start_date} to {end_date} ({duration_days} days)")
        sim_log.info(f"- {ctx.dim.nodes} geo nodes")

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: OnTick) -> None:
        sim_log.info("Completed simulation tick %d", tick.tick_index)

    def on_end(_: None) -> None:
        sim_log.info('Complete.')
        end_time = perf_counter()
        if start_time is not None:
            sim_log.info(f"Runtime: {(end_time - start_time):.3f}s")

    def adrio_start(adrio: AdrioStart) -> None:
        geo_log.debug(
            "Uncached geo attribute requested: %s. Retreiving now.", adrio.attribute)

    def on_movement_start(e: OnMovementStart) -> None:
        mm_log.info("Processing movement for day %d, step %d.", e.day, e.step)

    def on_movement_clause(e: OnMovementClause) -> None:
        cl_log = mm_log.getChild(e.clause_name)
        if e.total > 0:
            cl_log.debug("requested:\n%s", e.requested)
            if e.is_throttled:
                cl_log.debug(
                    "WARNING: movement is throttled due to insufficient population")
            cl_log.debug("moved:\n%s", e.actual)
        cl_log.info("moved %d individuals", e.total)

    def on_movement_finish(e: OnMovementFinish) -> None:
        mm_log.info(f"Moved a total of {e.total} individuals.")

    # Set up a subscriptions context, subscribe our handlers,
    # then yield to the outer context (where the sim should be run).
    with subscriptions() as subs:
        # Simulation logging
        subs.subscribe(sim.on_start, on_start)
        subs.subscribe(sim.on_tick, on_tick)
        subs.subscribe(sim.on_end, on_end)

        # Geo logging will be attached if it makes sense.
        sim_geo = getattr(sim, 'geo', None)
        if isinstance(sim_geo, DynamicGeoEvents):
            geo_log.info("Geo not loaded from cache; "
                         "attributes will be lazily loaded during simulation run.")
            subs.subscribe(sim_geo.adrio_start, adrio_start)

        # Movement logging
        subs.subscribe(sim.on_movement_start, on_movement_start)
        subs.subscribe(sim.on_movement_clause, on_movement_clause)
        subs.subscribe(sim.on_movement_finish, on_movement_finish)

        yield  # to outer context

    # Close out the log file.
    # This isn't necessary if we're running on the CLI, but if we're in a Jupyter context,
    # running the sim multiple times would keep appending to the file.
    # For most use-cases, just having one sim run in the log file is preferable.
    epy_log.removeHandler(log_handler)
    epy_log.setLevel(NOTSET)
