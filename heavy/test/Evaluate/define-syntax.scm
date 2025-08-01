; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

(define ok 'ok!)
(define-syntax my-lambda
  (syntax-rules (=>)
    ((my-lambda formals => body)
     (lambda formals
        body
        (write ok)
        (write "lambda args#:")
        (write (length 'formals))))))

; CHECK: (42 oops!)ok!"lambda args#:"1
((lambda (ok)
  ((my-lambda (x) => (write (list x ok))) 42))
 'oops!)
(newline)

;; TODO Local let-syntax needs to be supported.

#|
(define-syntax my-lambda
  (syntax-rules (:)
    ((my-lambda (arg ...) body ...)
     (lambda (arg ...)
       (write (string-append 'arg " has type " 'type)) ...
       body ...))))
((my-lambda ((x) (y))
  (write x)
  (write y)))
|#
