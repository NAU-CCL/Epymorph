"""
Implements the `verify` subcommand executed from __main__.
"""
import numpy as np

# Exit codes:
# - 0 verification success
# - 1 verification failed
# - 2 error loading files


def verify(output_file: str) -> int:
    """
    CLI command handler: verify a simulation output.
    Basically checks to make sure that, given compartmentalized populations at time zero
    along with the time-series of events affecting the populations, this checks to verify that
    the time-series of population by compartment has been calculated correctly.
    """

    # Currently this only works for SIRS output. However it should be possible
    # to extend this to work for any CompartmentModel, if given the model as well.

    # open output csv file
    try:
        with open(output_file, 'r', encoding='utf-8') as out_file:
            next(out_file)
            data = np.loadtxt(out_file, dtype=np.int64, delimiter=',')
    except Exception as e:
        print(f"Unable to read output file: {e}")
        return 2

    # determine number of populations and population totals using first row data
    pop_total = 0
    populations = []
    while (data[pop_total][1] < data[pop_total + 1][1]):
        current_population = data[pop_total][2] + \
            data[pop_total][3] + data[pop_total][4]
        populations.append(current_population)
        pop_total = pop_total + 1

    current_population = data[pop_total][2] + \
        data[pop_total][3] + data[pop_total][4]
    populations.append(current_population)
    pop_total = pop_total + 1

    time_steps = int(data.shape[0] / pop_total)

    print(
        f"Running verification assuming {pop_total} populations with totals {populations}")

    print()
    print("Compartment validation")

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
        return 1
    return 0
