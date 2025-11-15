## Init the reposiotry
conda env create -f environment.yml
conda activate my_env

## Update the yaml
bash: conda env export --no-builds | grep -v "^prefix:" > environment.yml
