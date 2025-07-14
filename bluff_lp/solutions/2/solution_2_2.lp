\ SCIP STATISTICS
\   Problem name     : model
\   Variables        : 16 (0 binary, 0 integer, 0 implicit integer, 16 continuous)
\   Constraints      : 14
Maximize
 Obj: -1 q[0]
Subject to
 c1: -1 q[0] +1 q[1] <= +0
 c2: +1 x[2] -1 x[4] -1 q[1] <= +0
 c3: -1 q[1] <= +0
 c4: -1 x[5] -1 x[8] -1 q[1] +1 q[2] <= +0
 c5: -1 q[2] <= +0
 c6: +1 x[8] -1 q[1] +1 q[3] <= +0
 c7: +1 x[10] -1 q[3] <= +0
 c8: -1 q[3] <= +0
 c9: +1 x[0] = +1
 c10: -1 x[0] +1 x[1] +1 x[2] +1 x[4] +1 x[7] = +0
 c11: -1 x[2] +1 x[3] = +0
 c12: -1 x[4] +1 x[5] +1 x[6] = +0
 c13: -1 x[7] +1 x[8] +1 x[9] +1 x[10] = +0
 c14: -1 x[10] +1 x[11] = +0
Bounds
 0 <= x[0] <= 1
 0 <= x[1] <= 1
 0 <= x[2] <= 1
 0 <= x[3] <= 1
 0 <= x[4] <= 1
 0 <= x[5] <= 1
 0 <= x[6] <= 1
 0 <= x[7] <= 1
 0 <= x[8] <= 1
 0 <= x[9] <= 1
 0 <= x[10] <= 1
 0 <= x[11] <= 1
 q[0] free
 q[1] free
 q[2] free
 q[3] free
End
