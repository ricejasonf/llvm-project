; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy base)
        (heavy builtins))
;       ^ Test importing symbols twice.

(define (check-list . xs)
  (write xs)
  (newline))

; CHECK: ()
(apply check-list)
; CHECK: (1)
(apply check-list 1)
; CHECK: (1 2)
(apply check-list 1 2)
; CHECK: (1 2 3)
(apply check-list 1 2 3)
; CHECK: (1 2 3 4 5)
(apply check-list 1 2 3 4 5)
; CHECK: (1 2 3 4 5)
(apply check-list 1 2 3 '(4 5))
; CHECK: (1 2 3 4 5)
(apply check-list 1 2 3 '(4 . 5))
