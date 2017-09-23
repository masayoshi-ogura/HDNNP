unset logscale y
stats output every ::50 nooutput
set logscale y
plot output index STATS_blocks-1 us 1:3 w l title 'energy RMSE', \
     output index STATS_blocks-1 us 1:4 w l title 'force RMSE', \
     output index STATS_blocks-1 us 1:5 w l title 'total RMSE', \

pause 5
reread
