; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

(define ok 'ok!)
(define-syntax my-lambda
  (syntax-rules (=>)
    ((my-lambda formals => body)
     (lambda formals body (write ok)))))

; CHECK: (42 oops!)ok!
((lambda (ok)
  ((my-lambda (x) => (write (list x ok))) 42))
 'oops!)
(newline)

;; TODO Local let-syntax needs to be supported.
