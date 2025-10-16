; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy base))

; CHECK: PASS
(write
  (cond
    ((> 1 0) 'PASS)))
(newline)

; CHECK: PASS
(write
  (cond
    ((not (> 0 1)) 'PASS)))
(newline)

; CHECK-NEXT: #<Undefined
(write
  (cond
    ((> 1 2) FAIL)))
(newline)

; CHECK-NEXT: PASS
(write
  (cond
    ((> 1 2) FAIL)
    (else 'PASS)))
(newline)

; CHECK-NEXT: PASS
(write
  (cond
    ((> 1 0) 'PASS)
    ((> 1 3) 'FAIL)
    ((> 1 0) 'FAIL)))
(newline)

; CHECK-NEXT: PASS
(write
  (cond
    ((> 1 2) 'FAIL)
    ((> 1 3) 'FAIL)
    ((> 1 0) 'PASS)))
(newline)

