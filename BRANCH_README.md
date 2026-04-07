# exp/fn-sim

This is an experiment to create a simulator with a more "functional programming" approach
to running epymorph simulations.

The thought was this might be more useful in situations
where a pipeline is running many independent simulations with the same RUME. If we could
effectively perform the "setup" overhead once as opposed to N times, perhaps that would
lead to some appreciable runtime savings. After this implementation I attempted to measure
said savings and determined it was actually pretty negligible. It seems my initial
measurements were somewhat confounded and rather than eliminating much processing time,
I only really managed to shift it into other components. Although this did not turn out
to be a performance win, I do like the structure of the code so I'm keeping this branch
for later consideration.

See:

- epymorph.simulator.basic.fn_simulator
- the accompanying pytest
