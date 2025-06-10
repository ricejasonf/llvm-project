; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

; CHECK: foo
(write (if #t 'foo 'bar))
(newline)

; CHECK: bar
(write (if #f 'foo 'bar))
(newline)

; CHECK-NEXT: foo
(if #t
  (write 'foo)
  (write 'bar))
(newline)

; CHECK-NEXT: bar
(if #f
  (write 'foo)
  (write 'bar))
(newline)

; CHECK-NEXT: foo
(if (null? '())
  (write 'foo)
  (write 'bar))
(newline)

; CHECK-NEXT: bar
(if (null? 42)
  (write 'foo)
  (write 'bar))
(newline)

(define fn
  ((lambda()
    (define captured-foo 'foo)
    (define captured-bar 'bar)
    (lambda (x)
      (if (null? x)
        (write captured-foo)
        (write captured-bar))))))

; CHECK-NEXT: foo
(fn '())
(newline)

; CHECK-NEXT: bar
(fn 42)
(newline)
