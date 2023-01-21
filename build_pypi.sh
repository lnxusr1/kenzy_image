#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd $SCRIPT_DIR

pytest $SCRIPT_DIR
if [ $? != 0 ]; then
    echo "\n"
    echo "CRITICAL: Unit Tests failed.  Unable to continue.\n"
    exit 1
fi

$SCRIPT_DIR/cleanup.sh
rm $SCRIPT_DIR/dist/*.*
rm -R -f $SCRIPT_DIR/build/*
python3 setup.py sdist bdist_wheel
twine upload --repository pypi dist/*
