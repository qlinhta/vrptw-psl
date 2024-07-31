#!/bin/bash

CUSTOMERS=50
ITERATIONS=10
LARGE_CONSTANT=10000

FILES=("R101" "R102" "R103" "R104" "R105" "C101" "C102" "C103" "C104" "C105" "RC101" "RC102" "RC103" "RC104" "RC105")

echo "Running heuristic solution..."
python3 ./src/insertion.py --files "${FILES[@]}" --customers $CUSTOMERS --iterations $ITERATIONS

echo "Running exact solution..."
python3 ./src/exact.py --files "${FILES[@]}" --customers $CUSTOMERS --large_constant $LARGE_CONSTANT

echo "All tasks completed."
