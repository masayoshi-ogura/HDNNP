# !!! required gnuplot version is 5.0 !!!
# Usage: gnuplot plot.gnu -c all|live
#        all ... plot all of blocks.
#        live ... plot latest block.

output = "`ls | grep .out | tail -n 1`"
# every ::50はwarning避け
stats output every ::50 nooutput
set logscale y
set format y "10^{%L}"
set yrange [1e-3:1e2]
set xlabel 'iteration'
set ylabel 'eV, eV/angstrom'

if (ARG1 eq 'all') {
  do for [i = 1:STATS_blocks-1] {
    set term x11 i
    plot output index i us 1:3 w l title 'energy RMSE', \
         output index i us 1:4 w l title 'force RMSE', \
         output index i us 1:5 w l title 'total RMSE'
  }
  pause -1 'Hit Return-Key to continue.'
}
if (ARG1 eq 'live') {
  load 'liveplot.gnu'
}
