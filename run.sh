#!/bin/bash

if [ $1 == "" ]; then
	echo "Usage: run.sh <algorithm name>"
fi

mvn exec:java -Dexec.mainClass="uk.ac.soton.ecs.imageclassifer.$1" -Dexec.args="imagesets/training imagesets/testing"
