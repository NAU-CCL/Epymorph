For your convenience, there are a number of ways to construct `TimeFrame` instances:

``` python
from datetime import date
from epymorph.time import TimeFrame

# 1. Start January 1st, 2020 and go for 150 days
TimeFrame(date(2020, 1, 1), 150)

# 2. Equivalent, but accept dates as ISO-8601 strings
TimeFrame.of("2020-01-01", 150)

# 3. January through March 2020
TimeFrame.range("2020-01-01", "2020-03-31")

# 4. Equivalent, but using an exclusive endpoint
TimeFrame.rangex("2020-01-01", "2020-04-01")

# 5. The whole of 2020
TimeFrame.year(2020)
```
