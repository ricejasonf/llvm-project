; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

; CHECK: foobar
(define fn
  ((lambda()
    (define captured-foo 'foo)
    (define captured-bar 'bar)
    (lambda ()
      (write captured-foo)
      (write captured-bar)))))
(fn)(newline)

(define global -1)

(set! global 1)

; CHECK: 1
(write global)
(newline)

; CHECK: 5
; CHECK-NEXT: 25
; CHECK-NEXT: 125
; CHECK-NEXT: 625
; CHECK-NEXT: 3125
; CHECK-NEXT: "Finally: "3125
(define (make-func x)
  (lambda (y)
    (* x y)))

(write
  ((lambda ()
    (define func (make-func 5))
    (define msg "")
    (set! global (func global))
    (write global)(newline)
    (set! global (func global))
    (write global)(newline)
    (set! global (func global))
    (write global)(newline)
    (set! global (func global))
    (write global)(newline)
    (set! global (func global))
    (write global)(newline)
    (set! msg "Finally: ")
    msg)))

(write global)
(newline)
