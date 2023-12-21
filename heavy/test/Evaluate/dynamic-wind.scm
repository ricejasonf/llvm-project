; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

; CHECK: "before"
; CHECK-NEXT: "during"
; CHECK-NEXT: "after"
(dynamic-wind
  (lambda ()
    (write "before")
    (newline))
  (lambda ()
    (write "during")
    (newline))
  (lambda ()
    (write "after")
    (newline))
)

(define-syntax inc
  (syntax-rules ()
    ((inc input) (set! input (+ input 1)))))
(define counter-enter 0)
(define counter-exit 0)
(define counter-within 0)
(define enter (lambda ()
  (write "fail")
  (newline)))
; CHECK: enter-environment-1
; CHECK: within-environment-1
; CHECK: exit-environment-1
(dynamic-wind
  (lambda ()
    (inc counter-enter)
    (write 'enter-environment-)
    (write counter-enter)
    (newline))
  (lambda ()
    (call/cc (lambda (cont)
      (set! enter cont)))
    (inc counter-within)
    (write 'within-environment-)
    (write counter-within)
    (newline))
  (lambda ()
    (inc counter-exit)
    (write 'exit-environment-)
    (write counter-exit)
    (newline)))

; CHECK: enter-environment-2
; CHECK: within-environment-2
; CHECK: exit-environment-2
; FIXME These escape procedures should not take an argument.
(enter 0)
; CHECK: enter-environment-3
; CHECK: within-environment-3
; CHECK: exit-environment-3
(enter 0)
; CHECK: enter-environment-4
; CHECK: within-environment-4
; CHECK: exit-environment-4
(enter 0)
