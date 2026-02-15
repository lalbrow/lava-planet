#!/bin/bash
#SBATCH --job-name=lava_planet
#SBATCH -p fdr
#SBATCH -n 16
#SBATCH --exclusive
#SBATCH --time=150:00:00
#SBATCH --mem=32G

# === Parse argument ===
BUILD_NAME=${1:-lava_build}    # use first argument or default to "lava_build"

# === Dynamic output naming ===
# SLURM can't expand shell vars in #SBATCH lines, so redirect manually
exec > ${BUILD_NAME}.out 2> ${BUILD_NAME}.err

echo "=== Lava Planet Run ==="
echo "Build name: ${BUILD_NAME}"
echo "Submitted from: $(pwd)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "========================="

# # === Environment setup ===
cd ~
source /usr/share/Modules/init/bash
source set_env.sh

cd ~/yz-canoe/canoe

# # === Safety check for existing build directory ===
if [ -d "${BUILD_NAME}" ]; then
    echo "Error: Directory '${BUILD_NAME}' already exists in $(pwd)."
    echo "Please choose a different build name or remove the existing directory."
    exit 1
fi

# === Build and run ===
mkdir -p ${BUILD_NAME}
cd ${BUILD_NAME}

cmake .. -DTASK=lava_planet
make -j 10

cd bin

# mpirun --mca plm_rsh_agent false ./lava_planet.release -i lava_spherical.inp
mpirun -n 16 ./lava_planet.release -i lava_spherical.inp
./combine.py -o test

cd ../..
python analyse_sph.py ${BUILD_NAME}/bin/lava_planet-test-main.nc
