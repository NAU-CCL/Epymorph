If we want to load data from a hypothetical SODA data source at `data.example.com` 
with ID `abcd-1234`, we could construct and execute a query as follows using
the classes and functions of this module:

``` python
import epymorph.adrio.soda as q

resource = q.SocrataResource(domain="data.example.com", id="abcd-1234")

query = q.Query(
    select=(
        q.Select("collection_week", dtype="date", as_name="date"),
        q.Select("fips_code", dtype="str", as_name="fips"),
        q.Select("patients_hospitalized", dtype="int", as_name="value"),
    ),
    where=q.And(
        q.DateBetween(
            "collection_week",
            date(2020, 1, 1),
            date(2020, 12, 31),
        ),
        q.In("fips_code", ["04013", "04005"]),
    ),
    order_by=(
        q.Ascending("collection_week"),
        q.Ascending("fips_code"),
        q.Ascending(":id"), # (1)!
    ),
)

result = q.query_csv(resource=resource, query=query)
```

1.  It's important that your query returns results in a _stable_ order &mdash; that is,
    if you repeated the query the results would be in the same order. This is needed so
    that pagination works, if required. Including `:id` in the order clause is a good way
    to guarantee this.
