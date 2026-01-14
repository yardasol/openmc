#! ~/bin/bash
set -e

sed -i "s/convert = False/convert = True/g" generate.py
python generate.py
openmc
sed -i "s/convert = True/convert = False/g" generate.py
sed -i "s/weight_windows = False/weight_windows = True/g" generate.py
python generate.py
openmc
mv statepoint.300.h5 statepoint.300_ww.h5
mv tallies.out tallies_ww.out
sed -i "s/weight_windows = True/weight_windows = False/g" generate.py
python generate.py
openmc


