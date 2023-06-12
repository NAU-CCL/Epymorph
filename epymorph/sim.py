from epymorph.data import geo_library, ipm, ipm_library, mm_library
from epymorph.run import run


def list_ipms() -> None:
    ipm_names = [ipm_name for ipm_name in ipm_library.keys()]
    print("Preset IPMs: ")
    for i, ipm_name in enumerate(ipm_names):
        print(f'{i+1}. {ipm_name}')


def list_mms() -> None:
    mm_names = [mm_name for mm_name in mm_library.keys()]
    print("Preset MMs: ")
    for i, mm_name in enumerate(mm_names):
        print(f'{i+1}. {mm_name}')


def list_geos() -> None:
    geos = [geo for geo in geo_library.keys()]
    print("Preset Geos: ")
    for i, geo_name in enumerate(geos):
        print(f'{i+1}. {geo_name}')


def run_sim():
    list_ipms()
    ipm_choice = list(ipm_library.keys())[int(
        input("\nEnter index of the IPM you would like to use: ")) - 1]
    list_mms()
    mm_choice = list(mm_library.keys())[int(
        input("\nEnter index of the MM you would like to use: ")) - 1]
    list_geos()
    geo_choice = list(geo_library.keys())[int(
        input("\nEnter index of the Geo you would like to use: ")) - 1]
    param_path = input("\nEnter path to your params TOML file: ")
    start_date_choice = input("\nEnter start date of simulation (YY-MM-DD): ")
    duration_choice = input(
        "\nEnter the duration of the simulation (d, m, y): ")
    run(ipm_name=ipm_choice,
        mm_name=mm_choice,
        geo_name=geo_choice,
        params_path=param_path,
        start_date_str=start_date_choice,
        duration_str=duration_choice,
        profiling=False,
        chart="e0",
        out_path=None)
