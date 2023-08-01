#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo 'Run pylint'
conda run -n survnet2-env pylint --rcfile=${DIR}/.pylintrc ${DIR}