unset logscale y
set yrange [*:*]
stats output every ::30 nooutput
set logscale y
set yrange [1e-3:1e2]
plot output index STATS_blocks-1 us 1:3 w l title 'energy RMSE', \
     output index STATS_blocks-1 us 1:4 w l title 'force RMSE', \
     output index STATS_blocks-1 us 1:5 w l title 'total RMSE', \

pause 5
reread
