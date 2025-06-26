"""Logging epymorph simulations to a file."""

from contextlib import contextmanager
from logging import BASIC_FORMAT, DEBUG, NOTSET, FileHandler, Formatter, getLogger
from time import perf_counter
from typing import Generator

from epymorph.event import (
    ADRIOProgress,
    EventBus,
    OnMovementClause,
    OnMovementFinish,
    OnMovementStart,
    OnStart,
    OnTick,
)
from epymorph.util import subscriptions

_events = EventBus()


@contextmanager
def file_log(
    log_file: str = "debug.log",
    log_level: str | int = DEBUG,
) -> Generator[None, None, None]:
    """
    Run simulations in this context manager to write detailed simulation activity
    to a log file.

    Parameters
    ----------
    log_file :
        The path to the log file to write. Can be relative to the current working
        directory.
    log_level :
        The log level to use; accepts any argument valid for
        [logging.Handler.setLevel][].

    Examples
    --------
    ```python
    with file_log("my_log_file.log"):
        sim = BasicSimulator(rume)
        my_results = sim.run()
    ```
    """

    # Initialize the logging system and create some Loggers for epymorph subsystems.
    log_handler = FileHandler(log_file, "w", "utf8")
    log_handler.setFormatter(Formatter(BASIC_FORMAT))

    epy_log = getLogger("epymorph")
    epy_log.addHandler(log_handler)
    epy_log.setLevel(log_level)

    sim_log = epy_log.getChild("sim")
    adrio_log = epy_log.getChild("adrio")
    mm_log = epy_log.getChild("movement")

    # Define handlers for each of the events we're interested in.

    start_time: float | None = None

    def on_start(e: OnStart) -> None:
        start_date = e.rume.time_frame.start_date
        end_date = e.rume.time_frame.end_date
        duration_days = e.rume.time_frame.days

        sim_log.info(f"Running simulation ({e.simulator}):")
        sim_log.info(f"- {start_date} to {end_date} ({duration_days} days)")
        sim_log.info(f"- {e.rume.scope.nodes} geo nodes")

        nonlocal start_time
        start_time = perf_counter()

    def on_tick(tick: OnTick) -> None:
        sim_log.info("Completed simulation tick %d", tick.tick_index)

    def on_finish(_: None) -> None:
        sim_log.info("Complete.")
        end_time = perf_counter()
        if start_time is not None:
            sim_log.info(f"Runtime: {(end_time - start_time):.3f}s")

    def on_adrio_progress(e: ADRIOProgress) -> None:
        if e.final:
            adrio_log.info(f"Loaded ADRIO {e.adrio_name} in ({e.duration:.3f} seconds)")

    def on_movement_start(e: OnMovementStart) -> None:
        mm_log.info("Processing movement for day %d, step %d.", e.day, e.step)

    def on_movement_clause(e: OnMovementClause) -> None:
        cl_log = mm_log.getChild(e.clause_name)
        if e.total > 0:
            cl_log.debug("requested:\n%s", e.requested)
            if e.is_throttled:
                cl_log.debug(
                    "WARNING: movement is throttled due to insufficient population"
                )
            cl_log.debug("moved:\n%s", e.actual.sum(axis=2))
        cl_log.info("moved %d individuals", e.total)

    def on_movement_finish(e: OnMovementFinish) -> None:
        mm_log.info(f"Moved a total of {e.total} individuals.")

    with subscriptions() as subs:
        # Set up a subscriptions context, subscribe our handlers,
        # then yield to the outer context (where the sim should be run).
        subs.subscribe(_events.on_start, on_start)
        subs.subscribe(_events.on_tick, on_tick)
        subs.subscribe(_events.on_finish, on_finish)

        subs.subscribe(_events.on_adrio_progress, on_adrio_progress)

        subs.subscribe(_events.on_movement_start, on_movement_start)
        subs.subscribe(_events.on_movement_clause, on_movement_clause)
        subs.subscribe(_events.on_movement_finish, on_movement_finish)

        yield  # to outer context
        # And now our event handlers will be unsubscribed.

    # Close out the log file.
    # This isn't necessary if we're running on the CLI, but if we're in a
    # Jupyter context, running the sim multiple times would keep appending to the file.
    # For most use-cases, just having one sim run in the log file is preferable.
    epy_log.removeHandler(log_handler)
    epy_log.setLevel(NOTSET)
