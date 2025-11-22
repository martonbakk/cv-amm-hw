## Init the reposiotry
conda env create -f environment.yml
conda activate cv-amm-hw

## Update the yaml
bash: conda env export --no-builds | grep -v "^prefix:" > environment.yml
