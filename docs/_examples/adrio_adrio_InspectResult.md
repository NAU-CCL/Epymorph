The simplest way to use an `InspectResult` is to print it!

``` python-console
>>> from epymorph.adrio import cdc
>>> from epymorph.kit import *
>>> result = (
...     cdc.COVIDFacilityHospitalization()
...     .with_context(
...         scope=CountyScope.in_states(["AZ"], year=2019),
...         time_frame=TimeFrame.rangex("2021-01-01", "2021-02-01"),
...     )
...     .inspect()
... )
>>> print(result)
ADRIO inspection for epymorph.adrio.cdc.COVIDFacilityHospitalization:
  Result shape: AxN (5, 15); dtype: date/value (int64); size: 75
  Date range: 2021-01-03 to 2021-01-31, period: 7 days
  Values:
    histogram: 11 █▅▂▂▂▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁ 3065
    quartiles: 54.8, 203.5, 593.0 (IQR: 538.2)
    std dev: 487.8
    percent zero: 0.0%
    percent adult_redacted: 5.3%
    percent adult_missing: 6.7%
    percent pediatric_redacted: 26.7%
    percent pediatric_missing: 6.7%
    percent unmasked: 64.0%
```
