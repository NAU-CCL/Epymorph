#!/bin/bash
# 
# This script configures a scaffold in which to run epymorph simulations on Monsoon,
# NAU's high-performance computing cluster powered by Slurm. https://in.nau.edu/arc/
# The ins-and-outs of using Monsoon are important but beyond this help text,
# so follow Monsoon's training first!
#
# === Goal ===
#
# This script does the following:
# - clone the epymorph repository (/home/$USER/epymorph)
# - create a separate folder for simulation scripts (/home/$USER/epymorph-jobs)
# - set up a venv and install epymorph's dependencies to make it runnable
# - set up a .env configuration file -- with some placeholders for you to replace!
# - create a folder for outputs (/scratch/$USER/epymorph-jobs)
# - create an example Slurm batch job runner script (job.sh)
# - create an example epymorph simulation runner script (example-job.py)
#
# After running this script successfully, you will be able to launch an array batch job
# that runs 10 simulations in parallel and writes their results file, console output,
# and error output (which is hopefully empty) to the outputs directory identified by job ID.
#
# === GitHub Credentials (PAT) ===
#
# In order to clone the epymorph repo (because it is not currently public) you will first need
# to set up a Personal Access Token (PAT). https://github.com/settings/tokens?type=beta
# Here are some recommended settings when generating a PAT:
# - Expiration: can be up to a year out, which is fine
# - Resource Owner: NAU-CCL
# - Only select repositories; Epymorph
# - Repository permissions:
#   - Contents: Read-only
#   - Metadata: Read-only
# You will use that to authenticate when the script prompts you -- use your GitHub username
# as the username and the PAT itself as your password.
#
# New PATs may require NAU-CCL admin approval before they are active.
#
# === Why the two job files? (job.sh and example-job.py) ===
#
# The job.sh file configures Slurm job settings and executes Python with a proper environment.
# The example-job.py file is where the simulation is configured and run, and where the results
# are dealt with. The job.sh file is designed to be re-usable (maybe with small modifications)
# while the *.py file is designed to be specific to your simulation task.
#
# === Usage ===
#
# Once you're set up with Monsoon access and logged in, copy this script into your
# home directory and make it runnable:
#
#   chmod +x ./setup-epymorph-monsoon.sh
#
# Then run it:
#
#   ./setup-epymorph-monsoon.sh
#
# === Handy Info ===
#
# Read more about which configuration settings are available for starting Slurm jobs:
#   https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS
#
# Read more about which environment variables are made available inside your Slurm job:
#   https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES
#
set -euo pipefail

dir=~
echo "Setting up epymorph project in $dir"
cd $dir

# Clone epymorph repo

if [ ! -d "epymorph" ]; then
    echo ""
    echo "You will first need to set up a PAT with read-access to the repo contents & metadata."
    echo "  https://github.com/settings/tokens?type=beta"
    echo "  (You may need admin approval before the token is active.)"
    echo "Provide your user name and use the PAT as your password when prompted..."
    echo ""

    git clone https://github.com/NAU-CCL/Epymorph.git epymorph
fi

# Set up jobs folder and output space

out_dir=/scratch/$USER/epymorph-jobs
mkdir -p $out_dir

prj_dir=/home/$USER/epymorph-jobs
mkdir -p $prj_dir
cd $prj_dir

if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install ../epymorph
else
    source .venv/bin/activate
fi

# Create env file for secrets.

tee .env >/dev/null <<EOF
EPYMORPH_CACHE_PATH=/scratch/$USER/.cache/epymorph
CENSUS_API_TOKEN=changeme
EOF

# Create job runner shell script.

tee job.sh >/dev/null <<'EOF'
#!/bin/bash
#SBATCH --job-name=epymorph
#SBATCH --output=/scratch/%u/epymorph-jobs/job_%A/stdout_%a.txt
#SBATCH --error=/scratch/%u/epymorph-jobs/job_%A/stderr_%a.txt
#SBATCH --time=1:00
#SBATCH --mem=192
#SBATCH --cpus-per-task=1
#SBATCH --array=1-8
set -euo pipefail

cd ~/epymorph-jobs

# Run a Python script given as the first argument
if [ $# -eq 0 ]; then
    echo "Error: No Python script provided."
    echo "Usage: $0 <script.py>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: File not found: $1"
    echo "Usage: $0 <script.py>"
    exit 1
fi

source .venv/bin/activate

env_file=.env
if [ -f "$env_file" ]; then
    export $(cat "$env_file" | xargs)
    echo "loaded $env_file"
fi

echo "starting job: $(pwd)/$1"
echo "which python: $(which python) ($(python -V))"

python "$1"

deactivate
echo "job complete"
EOF


# Create example python job script.

tee example-job.py >/dev/null <<'EOF'
import os

import numpy as np

from epymorph.kit import *
from epymorph.data.pei import pei_scope, pei_time_frame, pei_params


def main():
    rume = SingleStrataRume.build(
        ipm=ipm.Pei(),
        mm=mm.Pei(),
        scope=pei_scope,
        init=init.SingleLocation(location=0, seed_size=1_000),
        time_frame=pei_time_frame,
        params={
            "theta": 0.1,
            "move_control": 0.9,
            "infection_duration": 4,
            "immunity_duration": 90,
            **pei_params,
        },
    )

    sim = BasicSimulator(rume)
    with sim_messaging(sim):
        out = sim.run()

    user = os.getenv("USER")
    job_id = os.getenv("SLURM_ARRAY_JOB_ID", "?")
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "?")

    folder = f"/scratch/{user}/epymorph-jobs/job_{job_id}"
    os.makedirs(folder, exist_ok=True)

    np.savez(
        f"{folder}/results_{task_id}.npz",
        prevalence=out.prevalence,
        incidence=out.incidence
    )


if __name__ == "__main__":
    main()
EOF

# Done!

if epymorph --help >/dev/null 2>&1; then
    echo ""
    echo "epymorph setup successful"
    echo "Project files installed at $prj_dir"
    echo "Output files will be written to $out_dir"
    echo ""
    echo "First edit the .env file to configure your CENSUS_API_TOKEN."
    echo "Then you can run a test job by executing the commands:"
    echo "  cd $prj_dir"
    echo "  sbatch job.sh example-job.py"
    deactivate
else
    echo ""
    echo "epymorph setup failed"
    deactivate
    exit 1
fi
