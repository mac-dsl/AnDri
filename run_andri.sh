#bash
cd "$(dirname "$0")"
######## updated 2026-10-08 ##########
export PYTHONNOUSERSITE=1
export NUMBA_DISABLE_CUDA=1
export LD_PRELOAD=/home/parkj182/bin/miniconda3/envs/andri/lib/libstdc++.so.6
######## updated 2026-10-08 ##########
nm_len=2
if [ "$1" = CATSv2 ]; then nm_len=3; fi
######## updated 2026-10-08 ##########
delta_max=10
if [ "$1" = NAB ] || [ "$1" = Environ ]; then delta_max=8; fi
/home/parkj182/bin/miniconda3/envs/andri/bin/python test_andri.py -data "$1" -nm_len "$nm_len" -normalize zero-mean -k 5 -linkage ward -max_W 20 -delta_max "$delta_max" -rmin 0.005 -step True -rollback True
