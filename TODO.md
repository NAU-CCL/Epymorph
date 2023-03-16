# TODO

- MM file syntax and parser
- IPM file syntax and parser
- GEO file syntax and parser
- Generalized parameter/data access system
    - consists of:
        - GEO data associated with a node or node-pair, and can be time-series
        - Simulation parameters (e.g., duration of infection, interventions which dampen movement), may be time-series
    - used in:
        - MM when calculating movers
        - IPM when calculating events
    - modules need to be able to fetch the parameters dynamically, and
    - verify that the parameters available satisfy their requirements
        - How strict? a module that is _able_ to use time-series data certainly could use time-constant data by ignoring the time dimension
        - (You can't generally go the other way... maybe we should just do time-series for everything implicitly.)
- Generalized IPM mechanism
    - `pei.py` hard-codes certain specifics about its model, but
    - we actually need to represent the linkage between states because
    - you can use this to protect against generating unrealistic event counts (e.g., more infections than there are susceptible people to infect)
    - Ajay's work is instructive here.
- Initialization of model compartments as a kind of simulation parameter (initial infection)
- Handling of "insufficient population" movement events
    - Currently we silently cancel movement for an entire node for one tick if it doesn't have enough people to satisfy the requested movement
    - this could be handled in a number of ways, potentially
    - but at the very least (for now) should generate log messages
- Output artifacts: there are lots of possibilities for storing outputs of a run
- Stress testing: how big can we get before performance becomes unacceptable? (Nodes, IPM compartments/events, time duration)

## Questions

- `pei.py`'s commuter modeling (cloning Frank's approach)
    - currently does (for node to all-other-nodes):
        - a binomial draw to randomize the number of movers, then
        - a multinomial draw (proportional to typical movement) to randomize their destinations
    - Is this substantially different from doing a node-to-node poisson draw?
    - (A node-to-node poisson draw is how it determines dispersers.)
- How should we deal with situations where a pre-computed dataset is more efficient?
    - This is the case with pei-style dispersers, where movement is based on an evolved commuter matrix.
- Maybe it would be better to precompute an NxNxT array of movement magnitude,
    - as long as infection dynamics do not influence movement.
    - To save computation, for example, during parameter fitting.
    - Although this only works if none of the parameters being fit are in the MM...
    - Or maybe you can fit these separately?
