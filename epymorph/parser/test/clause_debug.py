import ast
import typing

from epymorph.parser.move_clause import Daily, daily


def main():
    s = """\
[mtype: days=[M,W,F]; leave=1; duration=1d; return=2; function=
def f(t, src, dst):
    commuters = geo['commuters'][src, dst]
    return poisson(t * commuters)
]
"""
    r = daily.parse_string(s, parse_all=True)
    # print(r.dump())
    # print(r[0])

    clause = typing.cast(Daily, r[0])
    node = ast.parse(clause.f, '<string>', mode='exec')
    # code = compile(node, '<string>', mode='exec')

    print(node.body[0])
    fn: ast.FunctionDef = node.body[0]  # type: ignore
    print(len(fn.args.args))

    # gs: dict[str, typing.Any] = {
    #     'geo': {
    #         'commuters': np.array([[0, 1000], [2000, 0]]),
    #     },
    #     'poisson': np.random.poisson,
    # }
    # ls: dict[str, typing.Any] = {}
    # exec(code, gs, ls)

    # print(ls['f'](0.75, 0, 1))
    # print(ls['f'](0.75, 1, 0))


if __name__ == '__main__':
    main()
