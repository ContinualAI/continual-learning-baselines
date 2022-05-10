#!/bin/bash
# first mandatory argument with path to continual-learning-baselines repository
# second optional argument with the test name to run. If not provided, all tests will be run

cd $1
if [ $# -ne 2 ]; then
    python -m unittest
else
    python -m unittest $2
fi

result=$?
if [ $? -ne 0 ]; then
    exit 1 # 1 -> bad
fi
exit 0 # 0 -> good
