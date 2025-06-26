from epymorph.data.ipm.sirs import SIRS
from epymorph.tools.ipm_diagram import (
    construct_digraph,
    render_diagram,
    render_diagram_to_bytes,
)


def test_construct_digraph():
    with construct_digraph(SIRS()) as diagram:
        assert diagram is not None
        source = diagram.source
    assert "S -> I" in source
    assert "I -> R" in source
    assert "R -> S" in source
    assert "S -> R" not in source


def test_render_diagram_to_bytes():
    image = render_diagram_to_bytes(SIRS())
    assert image is not None
    assert len(image.getvalue()) > 0


def test_render_diagram_to_bytes_same_result():
    image1 = render_diagram_to_bytes(SIRS())
    image2 = render_diagram_to_bytes(SIRS())
    assert image1.getvalue() == image2.getvalue()


def test_render_diagram(tmp_path):
    tmp_file = tmp_path / "image.png"
    render_diagram(SIRS(), file=tmp_file)
    assert tmp_file.is_file()
    assert tmp_file.stat().st_size > 0
