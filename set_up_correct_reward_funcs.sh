#!/bin/bash
RETRO_LOC=`pip show gym-retro | grep Location`
RETRO_LOC=${RETRO_LOC:10}
cd $RETRO_LOC
cd retro/data/stable/SonicTheHedgehog-Genesis
mv contest.json scenario.json
cd ../SonicTheHedgehog2-Genesis
mv contest.json scenario.json
cd ../SonicAndKnuckles3-Genesis
mv contest.json scenario.json