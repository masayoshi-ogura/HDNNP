unset logscale y
set yrange [*:*]
stats output every ::10 nooutput
set logscale y
set yrange [1e-4:1e2]

set term x11 0
set title 'training data'
plot output us 1:4:-2 lc var w l title 'total RMSE'

set term x11 1
set title 'validation data'
plot output us 1:7:-2 lc var w l title 'total RMSE'

#set term x11 2
#set title 'compare total RMSE'
#plot output us 1:4:-2 lc var w l title 'training data', \
#     output us 1:7:-2 lc var w l title 'validation data',


pause 5
reread
