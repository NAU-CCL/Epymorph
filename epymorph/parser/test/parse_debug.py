from epymorph.parser.movement import movement_spec


def main():
    file = './data/pei.movement'
    m = movement_spec.parse_file(file)
    print(m)


if __name__ == '__main__':
    main()
