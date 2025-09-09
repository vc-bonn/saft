git submodule update --init --recursive

# patch nvdiffrecmc
cd code/nvdiffrecmc/
git apply ../../patch.diff