#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}"/..

echo 'Create conda environment'
conda env create -f environment.yml

echo 'Install package'
