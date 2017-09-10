output = "`ls | grep .out | tail -n 1`"

set title output
set xrange [*:*]
set yrange [0:1]
set xlabel 'iteration'
set ylabel 'eV, eV/angstrom'

plot output us 1:3 w l title 'energy RMSE', \
  output us 1:4 w l title 'force RMSE', \
  output us 1:5 w l title 'total RMSE'

pause 5
reread
