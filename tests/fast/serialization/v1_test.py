from datetime import date
from typing import Sequence

import numpy as np
import sympy

import epymorph.serialization.model as model
import epymorph.serialization.v1 as s
from epymorph.compartment_model import MultiStrataModelSymbols, edge
from epymorph.data_shape import Shapes
from epymorph.kit import init, ipm, mm
from epymorph.util import SaveParams


def _test_serialization(model_obj, ser_obj, *, model_equality=None, **context_kwargs):
    """Test `serialize` and `deserialize` for a matched pair of objects."""
    if model_equality is None:
        model_equality = lambda a, b: a == b  # noqa: E731
    ctx = s.Context(**context_kwargs)
    assert s.serialize(model_obj, ctx) == ser_obj
    assert model_equality(s.deserialize(ser_obj, ctx), model_obj)


def symbol_dict(symbol_names: str) -> dict[str, sympy.Symbol]:
    return {str(x): x for x in sympy.symbols(symbol_names, seq=True)}


def test_compartment_name():
    a1 = model.CompartmentName.parse("foo")
    b1 = s.CompartmentName(base="foo")
    _test_serialization(a1, b1)

    a2 = model.CompartmentName.parse("foo_bar")
    b2 = s.CompartmentName(base="foo", subscript="bar")
    _test_serialization(a2, b2)

    a3 = model.CompartmentName("foo", "bar", "baz")
    b3 = s.CompartmentName(base="foo", subscript="bar", strata="baz")
    _test_serialization(a3, b3)


def test_compartment_def():
    a1 = model.CompartmentDef(
        name=model.CompartmentName.parse("foo"),
        tags=["tag_one", "tag_two"],
        description="test compartment",
    )
    b1 = s.CompartmentDef(
        name=s.CompartmentName(base="foo"),
        tags=["tag_one", "tag_two"],
        description="test compartment",
    )
    _test_serialization(a1, b1)


def test_edge_name():
    a1 = model.EdgeName(
        compartment_from=model.CompartmentName.parse("foo"),
        compartment_to=model.CompartmentName.parse("foo_bar"),
    )
    b1 = s.EdgeName(
        compartment_from=s.CompartmentName(base="foo"),
        compartment_to=s.CompartmentName(base="foo", subscript="bar"),
    )
    _test_serialization(a1, b1)


def _create_edge_def_pair():
    [c1, c2] = sympy.symbols("c1 c2")

    e1 = model.EdgeDef(
        name=model.EdgeName(
            compartment_from=model.CompartmentName.parse("A"),
            compartment_to=model.CompartmentName.parse("B_b"),
        ),
        rate=(c1 + 42 * c2),
        compartment_from=sympy.Symbol("A"),
        compartment_to=sympy.Symbol("B_b"),
    )

    e2 = s.EdgeDef(
        name=s.EdgeName(
            compartment_from=s.CompartmentName(base="A"),
            compartment_to=s.CompartmentName(base="B", subscript="b"),
        ),
        rate="c1 + 42*c2",
    )

    return (e1, e2)


def test_edge_def():
    (e1, e2) = _create_edge_def_pair()
    _test_serialization(e1, e2, symbols=symbol_dict("A B_b c1 c2"))


def _create_fork_def_pair() -> tuple[model.ForkDef, s.ForkDef]:
    [c1, c2, c3] = sympy.symbols("c1 c2 c3")

    def model_edge(rate: sympy.Expr) -> model.EdgeDef:
        return model.EdgeDef(
            name=model.EdgeName(
                compartment_from=model.CompartmentName.parse("A"),
                compartment_to=model.CompartmentName.parse("B"),
            ),
            rate=rate,
            compartment_from=sympy.Symbol("A"),
            compartment_to=sympy.Symbol("B"),
        )

    f1 = model.fork(
        model_edge(c1 + 1),
        model_edge(c2 + 2),
        model_edge(c3 + 3),
    )

    def ser_edge(rate: str) -> s.EdgeDef:
        return s.EdgeDef(
            name=s.EdgeName(
                compartment_from=s.CompartmentName(base="A"),
                compartment_to=s.CompartmentName(base="B"),
            ),
            rate=rate,
        )

    f2 = s.ForkDef(
        edges=[
            ser_edge("c1 + 1"),
            ser_edge("c2 + 2"),
            ser_edge("c3 + 3"),
        ],
    )

    return (f1, f2)


def test_fork_def():
    (f1, f2) = _create_fork_def_pair()
    _test_serialization(f1, f2, symbols=symbol_dict("A B c1 c2 c3"))


def test_transition_def():
    (e1, e2) = _create_edge_def_pair()
    (f1, f2) = _create_fork_def_pair()
    _test_serialization([e1, f1], [e2, f2], symbols=symbol_dict("A B B_b c1 c2 c3"))


def _create_attribute_def_pairs():
    a1 = model.AttributeDef(
        name="foo",
        type=int,
        shape=Shapes.TxN,
        default_value=3,
        comment="foo comment",
    )

    a2 = s.AttributeDef(
        name="foo",
        type="int",
        shape="TxN",
        default_value=3,
        comment="foo comment",
    )

    b1 = model.AttributeDef(
        name="foo2",
        type=(("longitude", float), ("latitude", float)),
        shape=Shapes.N,
        default_value=(0.0, 0.0),
    )

    b2 = s.AttributeDef(
        name="foo2",
        type="(('longitude',float),('latitude',float))",
        shape="N",
        default_value=(0.0, 0.0),
        comment=None,
    )

    return (a1, a2), (b1, b2)


def test_attribute_def():
    (a1, a2), (b1, b2) = _create_attribute_def_pairs()
    _test_serialization(a1, a2)
    _test_serialization(b1, b2)


def test_timeframe():
    a1 = model.TimeFrame(date(2020, 3, 4), 31)
    a2 = s.TimeFrame(start_date=date(2020, 3, 4), duration_days=31)
    _test_serialization(a1, a2)


def test_custom_scope():
    a1 = model.CustomScope(["A", "B", "C"])
    a2 = s.CustomScope(nodes=["A", "B", "C"])
    _test_serialization(a1, a2)


def test_state_scope():
    a1 = model.StateScope.in_states(["AZ", "NM"], year=2020)
    a2 = s.StateScope(includes=["04", "35"], includes_granularity="state", year=2020)
    _test_serialization(a1, a2)


def test_county_scope():
    a1 = model.CountyScope.in_counties(["Maricopa, AZ", "Bernalillo, NM"], year=2019)
    a2 = s.CountyScope(
        includes=["04013", "35001"],
        includes_granularity="county",
        year=2019,
    )
    _test_serialization(a1, a2)


def test_tract_scope():
    a1 = model.TractScope.in_tracts(["04013114500", "04013114600"], year=2018)
    a2 = s.TractScope(
        includes=["04013114500", "04013114600"],
        includes_granularity="tract",
        year=2018,
    )
    _test_serialization(a1, a2)


def test_block_group_scope():
    a1 = model.BlockGroupScope.in_block_groups(
        ["040131145001", "040131145002"],
        year=2017,
    )
    a2 = s.BlockGroupScope(
        includes=["040131145001", "040131145002"],
        includes_granularity="block group",
        year=2017,
    )
    _test_serialization(a1, a2)


class Foo1:
    a: int

    def __init__(self):
        self.a = 42


def test_dynamic_class_no_args():
    a = Foo1()
    b = s.DynamicClass._serialize(a, s.Context())
    c = s.DynamicClass._deserialize(b, s.Context())

    expected = s.DynamicClass(classpath=f"{__name__}.Foo1", args=[], kwargs={})
    assert b == expected
    assert c.a == 42


class Foo2(SaveParams):
    a: int
    b: str

    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_dynamic_class_with_args():
    a = Foo2(42, b="b_value")
    b = s.DynamicClass._serialize(a, s.Context())
    c = s.DynamicClass._deserialize(b, s.Context())

    expected = s.DynamicClass(
        classpath=f"{__name__}.Foo2",
        args=[42],
        kwargs={"b": "b_value"},
    )
    assert b == expected
    assert c.a == 42


def _classpath(cls):
    return f"{cls.__module__}.{cls.__name__}"


def test_gpm():
    a1 = model.GPM(
        name="alpha",
        ipm=ipm.SIRS(),
        mm=mm.Centroids(),
        init=init.SingleLocation(0, seed_size=1000),
        params={model.ModuleNamePattern.parse("beta"): 0.4},
    )
    a2 = s.GPM(
        name="alpha",
        ipm=s.DynamicClass(
            classpath=_classpath(ipm.SIRS),
            args=[],
            kwargs={},
        ),
        mm=s.DynamicClass(
            classpath=_classpath(mm.Centroids),
            args=[],
            kwargs={},
        ),
        init=s.SimulationFunction(
            classpath=_classpath(init.SingleLocation),
            args=[0],
            kwargs={"seed_size": 1000},
        ),
        params={"*::beta": 0.4},
    )

    def init_equality(this, that):
        if type(this) is not type(that):
            return False
        this_args, this_kwargs = this.init_params
        that_args, that_kwargs = that.init_params
        return this_args == that_args and this_kwargs == that_kwargs

    def gpm_equality(this, that):
        return (
            this.name == that.name
            and type(this.ipm) is type(that.ipm)
            and type(this.mm) is type(that.mm)
            and init_equality(this.init, that.init)
            and this.params == that.params
        )

    _test_serialization(a1, a2, model_equality=gpm_equality)


def test_single_strata_rume():
    a = model.SingleStrataRUME.build(
        ipm=ipm.SIRS(),
        mm=mm.Centroids(),
        init=init.SingleLocation(0, seed_size=1000),
        scope=model.StateScope.in_states(["AZ", "NM", "CO"], year=2020),
        time_frame=model.TimeFrame.rangex("2020-01-01", "2021-01-01"),
        params={
            "beta": 0.4,
            "gamma": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        },
    )

    b = s.SingleStrataRUME(
        strata=[
            s.GPM(
                name="all",
                ipm=s.DynamicClass(
                    classpath=_classpath(ipm.SIRS),
                    args=[],
                    kwargs={},
                ),
                mm=s.DynamicClass(
                    classpath=_classpath(mm.Centroids),
                    args=[],
                    kwargs={},
                ),
                init=s.SimulationFunction(
                    classpath=_classpath(init.SingleLocation),
                    args=[0],
                    kwargs={"seed_size": 1000},
                ),
                params=None,
            )
        ],
        scope=s.StateScope(
            year=2020,
            includes_granularity="state",
            includes=["04", "08", "35"],
        ),
        time_frame=s.TimeFrame(start_date=date(2020, 1, 1), duration_days=366),
        params={
            "*::*::beta": 0.4,
            "*::*::gamma": s.NDArray(dtype="floating", values=[0.1, 0.2, 0.3]),
        },
    )

    assert s.serialize(a) == b
    s.deserialize(b)  # no error


def test_multi_strata_rume():
    def meta_edges(symbols: MultiStrataModelSymbols) -> Sequence[model.TransitionDef]:
        [s, i, _] = symbols.strata_compartments("alpha")
        [meta_param_a] = symbols.all_meta_requirements
        return [
            edge(s, i, rate=meta_param_a / 2),
        ]

    a = model.MultiStrataRUME.build(
        strata=[
            model.GPM(
                name="alpha",
                ipm=ipm.SIRS(),
                mm=mm.Centroids(),
                init=init.SingleLocation(0, seed_size=1000),
            ),
            model.GPM(
                name="beta",
                ipm=ipm.SIRH(),
                mm=mm.Flat(),
                init=init.NoInfection(),
            ),
        ],
        meta_requirements=[
            model.AttributeDef("meta_param_a", int, Shapes.N),
        ],
        meta_edges=meta_edges,
        scope=model.StateScope.in_states(["AZ", "NM", "CO"], year=2020),
        time_frame=model.TimeFrame.rangex("2020-01-01", "2021-01-01"),
        params={
            "beta": 0.4,
            "gamma": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        },
    )

    b = s.MultiStrataRUME(
        strata=[
            s.GPM(
                name="alpha",
                ipm=s.DynamicClass(
                    classpath=_classpath(ipm.SIRS),
                    args=[],
                    kwargs={},
                ),
                mm=s.DynamicClass(
                    classpath=_classpath(mm.Centroids),
                    args=[],
                    kwargs={},
                ),
                init=s.SimulationFunction(
                    classpath=_classpath(init.SingleLocation),
                    args=[0],
                    kwargs={"seed_size": 1000},
                ),
                params=None,
            ),
            s.GPM(
                name="beta",
                ipm=s.DynamicClass(
                    classpath=_classpath(ipm.SIRH),
                    args=[],
                    kwargs={},
                ),
                mm=s.DynamicClass(
                    classpath=_classpath(mm.Flat),
                    args=[],
                    kwargs={},
                ),
                init=s.SimulationFunction(
                    classpath=_classpath(init.NoInfection),
                    args=[],
                    kwargs={},
                ),
                params=None,
            ),
        ],
        meta_requirements=[
            s.AttributeDef(
                name="meta_param_a",
                type="int",
                shape="N",
                default_value=None,
                comment=None,
            ),
        ],
        meta_edges=[
            s.EdgeDef(
                name=s.EdgeName(
                    compartment_from=s.CompartmentName(
                        base="S",
                        subscript=None,
                        strata="alpha",
                    ),
                    compartment_to=s.CompartmentName(
                        base="I",
                        subscript=None,
                        strata="alpha",
                    ),
                ),
                rate="meta_param_a_meta/2",
            ),
        ],
        scope=s.StateScope(
            year=2020,
            includes_granularity="state",
            includes=["04", "08", "35"],
        ),
        time_frame=s.TimeFrame(start_date=date(2020, 1, 1), duration_days=366),
        params={
            "*::*::beta": 0.4,
            "*::*::gamma": s.NDArray(dtype="floating", values=[0.1, 0.2, 0.3]),
        },
    )

    assert s.serialize(a) == b
    s.deserialize(b)  # no error
