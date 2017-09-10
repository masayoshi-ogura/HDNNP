output = "`ls | grep .out | tail -n 1`"

set xrange [*:*]
set yrange [0:1]
plot output us 1:3 w l, output us 1:4 w l, output us 1:5 w l

pause 5
reread
