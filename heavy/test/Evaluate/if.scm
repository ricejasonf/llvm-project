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

; CHECK-NEXT: 0
((lambda ()
  (define temp 0)
  (if (eqv? temp 42)
    (set! temp 5))
  (write temp)))
(newline)

; CHECK-NEXT: 5
((lambda ()
  (define temp 42)
  (if (eqv? temp 42)
    (set! temp 5))
  (write temp)))
(newline)

; CHECK-NEXT: 5
((lambda ()
  (define temp 42)
  (if (eqv? temp 42)
    (set! temp (+ 5))
    (set! temp 6))
  (write temp)))
(newline)

; CHECK-NEXT: 6
((lambda ()
  (define temp 42)
  (if (eqv? temp 43)
    (set! temp 5)
    (set! temp (+ 5 1)))
  (write temp)))
(newline)

; CHECK-NEXT: foo
(write
  (if (null? '()) 'return-foo 'return-bar))
(newline)

; CHECK-NEXT: bar
(write
  (if (null? 42) 'return-foo 'return-bar))
(newline)

; CHECK-NEXT: baz
(write
  (if (null? '()) 'return-baz))
(newline)

; CHECK-NEXT: Undefined
(write
  (if (null? 42) 'return-baz))
(newline)
