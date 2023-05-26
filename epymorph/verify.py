import numpy as np

# "verify" subcommand
def verify(file_path) -> int:
    # open output csv file
    try:
        with open(file_path, 'r') as file:
            next(file)
            data = np.loadtxt(file, dtype=int, delimiter=',')
    except Exception as e:
        print(f"Unable to read output file: {e}")
        return 2
        
    print("Compartment validation")
        
    compartment_total = 100000

    comp_passed = True
    error_locations = []
    # loop through each row, adding all compartments and checking them against expected total
    for i in range(0, data.shape[0]):
        row_total = data[i][2] + data[i][3] + data[i][4]
        if row_total != compartment_total:
            comp_passed = False
            error_locations.append(i)
            
    # display result of compartment verification
    if comp_passed:
        print("[✓] Correct compartment totals")
    else:
        print(f"[✗] Incorrect compartment totals on time steps {error_locations}")
        
    print()
    print("Event validation")
       
    event_passed = True
    error_locations = []
    # prime loop by calculating initial compartment values
    c0 = data[0][2] + data[0][5] - data[0][7]
    c1 = data[0][3] - data[0][5] + data[0][6]
    c2 = data[0][4] - data[0][6] + data[0][7]
    expected_results = [0, 0, 0]

    # loop through each row, comparing compartment values to those of the previous row with event results applied
    for i in range(0, data.shape[0]):
        # calculate expected results using compartment values from previous row
        expected_results[0] = c0 - data[i][5] + data[i][7]
        expected_results[1] = c1 + data[i][5] - data[i][6]
        expected_results[2] = c2 + data[i][6] - data[i][7]
        
        # update compartment values to those of current row
        c0 = data[i][2]
        c1 = data[i][3]
        c2 = data[i][4]
        
        if c0 != expected_results[0] or c1 != expected_results[1] or c2 != expected_results[2]:
            event_passed = False
            error_locations.append(i)
        
    # display result of event validation
    if event_passed:
        print("[✓] Correct event results")
    else:
        print(f"[✗] Incorrect event results on time steps {error_locations}")
        
    # return exit code according to results
    if not comp_passed or not event_passed:
        return 3
    return 0
