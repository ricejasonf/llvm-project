; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines

0
; CHECK:0
0.0
; CHECK:0

1 2 3 10 19 1025
; CHECK:1
; CHECK:2
; CHECK:3
; CHECK:10
; CHECK:19
; CHECK:1025

; Support explicit sign.
+1 +2 +3 +10 +19 +1025
; CHECK:1
; CHECK:2
; CHECK:3
; CHECK:10
; CHECK:19
; CHECK:1025

-1 -2 -3 -10 -19 -1025
; CHECK:-1
; CHECK:-2
; CHECK:-3
; CHECK:-10
; CHECK:-19
; CHECK:-1025

1.024
; CHECK:1.02400{{[0-9]*}}
-1.024
; CHECK:-1.02400{{[0-9]*}}

.0051
; CHECK:0.005{{[0-9]*}}
-.0051
; CHECK:-0.005{{[0-9]*}}

.00051
; CHECK:5.{{[0-9]*}}e-4

-.00051
; CHECK:-5.{{[0-9]*}}e-4

5.0e-4
; CHECK:5.0{{[0-9]*}}e-4
5.e-4
; CHECK:5.0{{[0-9]*}}e-4
5e-4
; CHECK:5.0{{[0-9]*}}e-4
-5e-4
; CHECK:-5.0{{[0-9]*}}e-4

50000000.0
; CHECK:5.0{{[0-9]*}}e+7

5e7
; CHECK:5.0{{[0-9]*}}e+7
5e+7
; CHECK:5.0{{[0-9]*}}e+7

; Support radix prefix
#b0 #o0 #d0 #x0
; CHECK-COUNT-4:0

; Support exactness prefix
#e0 #i0
; CHECK-COUNT-2:0

; Support exactness/radix prefix
#e#b0 #e#o0 #e#d0 #e#x0
#i#b0 #i#o0 #i#d0 #i#x0
#b#e0 #o#e0 #d#e0 #x#e0
#b#i0 #o#i0 #d#i0 #x#i0
; CHECK-COUNT-16:0

#b1001
; CHECK:9

#d10
; CHECK:10

#o16
; CHECK:14

#xFF
; CHECK:255
#xff
; CHECK:255

#i5
; CHECK:5

#i5.0
; CHECK:5
