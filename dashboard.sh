#!/bin/bash
screen -d -m -S sonicDashboard 'visdom'
cd supersonic
python vis_plot.py -port "$1" -server "$2" -run "$3"
function close_screen {
    screen -X -S sonicDashboard quit
}
while true; do
    cd .
done
trap close_screen EXIT