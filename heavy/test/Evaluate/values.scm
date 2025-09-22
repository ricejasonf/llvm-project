; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

; CHECK: -1
(write
  (call-with-values * -))
(newline)

; CHECK: nothing
(call-with-values
  (lambda () (values))
  (lambda () (write 'nothing)))
(newline)

; CHECK-NEXT: 4
(write
  (call-with-values
    (lambda () (values 4 5))
    (lambda (a b) a)))
(newline)

; CHECK-NEXT: 5
(write
  (call-with-values
    (lambda () (values 4 5))
    (lambda (a b) b)))
(newline)

; CHECK-NEXT: 32
(write
  (call-with-values
    (lambda () (values 1 1 2 4 8 16))
    +))
(newline)
