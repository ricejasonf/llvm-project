; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

;; TODO Local let-syntax needs to be tested/supported.

(define-syntax capture-late-bind
  (syntax-rules ()
    ((capture-late-bind x)
     ((lambda ()
       (define storage 0)
       (lambda ()
         (set! storage
           (if (symbol? ok)
               ok 'oknotyet))
         (write storage)))))))

; CHECK: oknotyet
((capture-late-bind 5))
(newline)

(define ok 'ok!)

; CHECK-NEXT: ok
((capture-late-bind 5))

(newline)
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

; CHECK: 42"x has type Int""y has type Int"
; CHECK-NEXT: 5
; CHECK-NEXT: 6
(define-syntax my-lambda
  (syntax-rules (:)
    ((my-lambda ((arg : type) ...) body ...)
     (lambda (arg ...)
       (write 42)
       (write (string-append 'arg " has type " 'type)) ...
       (newline)
       body ...))))
((my-lambda ((x : Int) (y : Int))
  (write x)
  (newline)
  (write y)) 5 6)
(newline)

; CHECK: (0 1)
; CHECK-NEXT: (0 1 2)
(define-syntax ez
  (syntax-rules ()
    ((ez 0 1 i ...)
     '(0 1 i ...))))
(write (ez 0 1))
(newline)
(write (ez 0 1 2))
(newline)

; CHECK: (0 1 2 3 4 9)
(define-syntax ez
  (syntax-rules ()
    ((ez 0 1 i ... 5 6)
     '(0 1 i ... 9))))
(write (ez 0 1 2 3 4 5 6))
(newline)

(define-syntax my-define
  (syntax-rules ()
    ((my-define name x)
     (define name '(my name x)))))

; CHECK: (my my-tl 42)
(my-define my-tl 42)
(write my-tl)
(newline)

; CHECK: Undefined
((lambda ()
  (my-define not-my-local 12)
  (write not-my-local)
  (newline)))
