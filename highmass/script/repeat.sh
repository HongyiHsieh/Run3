#!/bin/bash

while ! source /eos/home-h/hhsieh/Run3/script/run_HiggsDNA.sh; do
    echo "Resarting the program..."
    sleep 5
done

echo "The program has finished running."
