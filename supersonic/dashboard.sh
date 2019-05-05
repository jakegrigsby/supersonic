#!/bin/bash
screen -d -m -S sonicDashboard 'visdom'
python vis_plot.py -run "$1"
function close_screen {
    screen -X -S sonicDashboard quit
}
while true; do
    cd .
done
trap close_screen EXIT
