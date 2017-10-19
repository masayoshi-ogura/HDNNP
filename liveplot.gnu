unset logscale y
set yrange [*:*]
stats output every ::30 nooutput
set logscale y
set yrange [1e-4:1e2]

set term x11 0
set title 'training data'
plot output index STATS_blocks-1 us 1:3 w l title 'RMSE', \
     output index STATS_blocks-1 us 1:4 w l title 'dRMSE', \
     output index STATS_blocks-1 us 1:5 w l title 'total RMSE'

set term x11 1
set title 'validation data'
plot output index STATS_blocks-1 us 1:6 w l title 'RMSE', \
     output index STATS_blocks-1 us 1:7 w l title 'dRMSE', \
     output index STATS_blocks-1 us 1:8 w l title 'total RMSE'

set term x11 2
set title 'compare total RMSE'
plot output index STATS_blocks-1 us 1:5 w l title 'training data', \
     output index STATS_blocks-1 us 1:8 w l title 'validation data',


pause 5
reread
