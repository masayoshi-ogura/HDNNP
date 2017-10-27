# !!! required gnuplot version is 5.0 !!!
# Usage: gnuplot plot.gnu -c all|live
#        all ... plot all of blocks.
#        live ... plot latest block.

output = "`find . -name *progress.dat | sort | tail -n 1 | head -n 1`"
print output
set xlabel 'iteration'
set ylabel 'eV, eV/angstrom'

if (ARG1 eq 'all') {
  # データ数が複雑になるので一旦保留

  # # every ::30はwarning避け
  # stats output every ::30 nooutput
  # do for [i = 1:STATS_blocks-1] {
  #   set term x11 i
  #   set logscale y
  #   set yrange [1e-3:1e2]
  #   plot output index i us 1:3 w l title 'RMSE', \
  #        output index i us 1:4 w l title 'dRMSE', \
  #        output index i us 1:5 w l title 'total RMSE'
  # }
  # pause -1 'Hit Return-Key to continue.'
}
if (ARG1 eq 'live') {
  load 'liveplot.gnu'
}
