# V7 Error Diagnostics Report

N=21737  RMSE=1.8767s  MAE=1.4620s  Bias=-1.0510s


## 1. Stint-position × compound

### SOFT
```
                rmse      bias       n
age_bucket                            
1-5         1.569525 -0.445717   553.0
6-10        1.301419 -0.330570   810.0
11-20       1.186561 -0.304662  1038.0
21-35       1.307663 -0.090887   366.0
36+         1.854577  0.759397     6.0
```

### MEDIUM
```
                rmse      bias       n
age_bucket                            
1-5         2.047143 -0.891387  1482.0
6-10        1.838281 -1.015925  2156.0
11-20       1.652757 -0.982584  3462.0
21-35       1.469320 -0.736873  1877.0
36+         1.961790 -0.904543   187.0
```

### HARD
```
                rmse      bias       n
age_bucket                            
1-5         2.139334 -1.443706  1037.0
6-10        2.086381 -1.325714  1672.0
11-20       2.056781 -1.369294  3233.0
21-35       2.077667 -1.325292  3051.0
36+         2.518987 -1.657793   807.0
```

## 2. Race lap × compound

### SOFT
```
                rmse      bias       n
lap_bucket                            
1-10        1.091850 -0.281227   596.0
11-20       0.940537 -0.277941   426.0
21-35       1.326357 -0.312007   247.0
36+         1.485790 -0.329861  1504.0
```

### MEDIUM
```
                rmse      bias       n
lap_bucket                            
1-10        1.802558 -1.042249  1777.0
11-20       1.658142 -0.944409  2123.0
21-35       1.479806 -0.671764  2226.0
36+         1.923469 -1.024667  3038.0
```

### HARD
```
                rmse      bias       n
lap_bucket                            
1-10        2.184155 -1.291682   603.0
11-20       2.106708 -1.446791  1153.0
21-35       2.062686 -1.354818  3029.0
36+         2.146694 -1.390064  5015.0
```

## 3. Outliers >3s
Total: 2265 (10.4%)

By circuit:
```
circuit_str
Canadian Grand Prix      1160
Monaco Grand Prix         235
Italian Grand Prix        176
Azerbaijan Grand Prix     144
São Paulo Grand Prix      127
Hungarian Grand Prix      112
Las Vegas Grand Prix       89
Singapore Grand Prix       41
Japanese Grand Prix        37
British Grand Prix         33
```

By compound:
```
               n  pct_of_outliers  median_tire_age
tire_str                                          
HARD      1515.0        66.887417             18.0
MEDIUM     680.0        30.022075             10.0
SOFT        70.0         3.090508              5.5
```

By age bucket:
```
age_bucket
1-5      405
6-10     440
11-20    701
21-35    480
36+      239
```

## 4. Within-stint residual autocorrelation
Mean |delta residual| between consecutive laps: 0.460s
Std of per-stint mean residual: 1.398s
  (If purely random noise: expected ~1.170s; higher = systematic per-stint offset)

## 5. Position & gap effects

By position bucket:
```
                rmse      bias       n
pos_bucket                            
P1-3        1.895529 -1.160307  3548.0
P4-8        1.875685 -1.071646  5881.0
P9-15       1.835783 -0.987186  8199.0
P16+        1.941541 -1.054617  4109.0
```

Dirty air flag:
```
                    rmse      bias        n
dirty_air_flag                             
0               1.894663 -1.080815  15157.0
1               1.834713 -0.982457   6580.0
```
Corr(gap_ahead, abs_error): 0.008
Corr(gap_leader, abs_error): -0.036