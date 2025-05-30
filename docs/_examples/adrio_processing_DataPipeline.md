This example uses a `DataPipeline` to process an simple `DataFrame`:

``` python
import numpy as np
import pandas as pd
from epymorph.adrio.processing import DataPipeline, Fill, PivotAxis, RandomFix
from epymorph.simulation import Context
from epymorph.kit import *

# Example data: integer values for each pair of 2 places and 3 variables.
raw_data_df = pd.DataFrame(
    {
        "geoid": ["04", "04", "04", "35", "35"],
        "variable": ["a", "b", "c", "a", "b"],
        "value": [11, -999, 13, 21, 22],
    }
)

# Usually we'd be doing this with a real simulation context.
context = Context.of(rng=np.random.default_rng(42))

# Define the pipeline...
pipeline = (
    DataPipeline(
        # `axes` defines the axes of the result array,
        # as well as the set of values that should be in each axis.
        axes=(
            PivotAxis("geoid", ["04", "35"]),  # first axis
            PivotAxis("variable", ["a", "b", "c"]),  # second axis
        ),
        ndims=2,
        dtype=np.int64,
        rng=context,
    )
    # Replace sentinel values (-999) with a random value from 1 to 3.
    .strip_sentinel(
        "insufficient_data",
        np.int64(-999),
        RandomFix.from_range(1, 3),
    )
    # Fill missing values with 0.
    .finalize(Fill.of_int64(0))
)

# Run the data through the pipeline.
result = pipeline(raw_data_df)

result.value
# array([[11,  1, 13],
#        [21, 22,  0]])
```
