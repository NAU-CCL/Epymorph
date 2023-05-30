import numpy as np

# "verify" subcommand


def verify(output_file, population_file) -> int:
    # open output csv file
    try:
        with open(output_file, 'r') as out_file:
            next(out_file)
            data = np.loadtxt(out_file, dtype=int, delimiter=',')
    except Exception as e:
        print(f"Unable to read output file: {e}")
        return 2

    print("Compartment validation")

    # open population csv file
    try:
        with open(population_file) as pop_file:
            # TODO: ignore leading comments
            # placeholder
            next(pop_file)
            next(pop_file)
            populations = np.loadtxt(pop_file, dtype=int)
    except Exception as e:
        print(f"Unable to read population file: {e}")
        return 2

    # upper bounds of nested for loops
    time_steps = int(data.shape[0] / populations.shape[0])
    pop_total = populations.shape[0]

    comp_passed = True
    error_locations = []
    index = 0
    # loop through each row, adding all compartments and checking them against expected total
    # i = time step, j = population
    for i in range(0, time_steps):
        for j in range(0, pop_total):
            row_total = data[index][2] + data[index][3] + data[index][4]
            if row_total != populations[j]:
                comp_passed = False
                error_locations.append((i, j))

            index = index + 1

    # display result of compartment verification
    if comp_passed:
        print("[✓] Correct compartment totals")
    else:
        print(f"[✗] Incorrect compartment totals on steps {error_locations}")

    print()
    print("Event validation")

    event_passed = True
    comp_data = []
    error_locations = []

    # prime loop by calculating initial compartment values for each population
    for i in range(0, pop_total):
        c0 = data[i][2] + data[i][5] - data[i][7]
        c1 = data[i][3] - data[i][5] + data[i][6]
        c2 = data[i][4] - data[i][6] + data[i][7]
        comp_data.append([c0, c1, c2])
    expected_results = [0, 0, 0]
    index = 0

    # loop through each row and population, comparing compartment values to those of the previous row for that population with event results applied
    for i in range(0, time_steps):
        for j in range(0, pop_total):
            # calculate expected results using compartment values from previous row
            expected_results[0] = comp_data[j][0] - \
                data[index][5] + data[index][7]
            expected_results[1] = comp_data[j][1] + \
                data[index][5] - data[index][6]
            expected_results[2] = comp_data[j][2] + \
                data[index][6] - data[index][7]

            # update compartment values to those of current row
            comp_data[j][0] = data[index][2]
            comp_data[j][1] = data[index][3]
            comp_data[j][2] = data[index][4]

            index = index + 1

            if comp_data[j][0] != expected_results[0] or comp_data[j][1] != expected_results[1] or comp_data[j][2] != expected_results[2]:
                event_passed = False
                error_locations.append((i, j))

    # display result of event validation
    if event_passed:
        print("[✓] Correct event results")
    else:
        print(f"[✗] Incorrect event results on time steps {error_locations}")

    # return exit code according to results
    if not comp_passed or not event_passed:
        return 3
    return 0
