; RUN: heavy-scheme %s | FileCheck %s
(import (heavy base))

; CHECK: (1 2 "yo")
(write (list 1 2 "yo"))(newline)

(define nums (list 1 2 3))
; CHECK: (1 2 3)
(write nums)(newline)
; CHECK-NEXT: (0 1 2 3)
(write (cons 0 nums))(newline)
; CHECK-NEXT: (1 2 3)
(write nums)(newline)
; CHECK-NEXT: 1
(write (car nums))(newline)
; CHECK-NEXT: (2 3)
(write (cdr nums))(newline)
; CHECK-NEXT: (1 2 3)
(write (append nums))(newline)
; CHECK-NEXT: (1 2 3)
(write (append nums '()))(newline)
; CHECK-NEXT: (1 2 3)
(write (append '() nums))(newline)
; CHECK-NEXT: (4 5 6 1 2 3)
(write (append '(4 5 6) nums))(newline)
; CHECK-NEXT: (1 2 3 4 5 6)
(write (append nums '(4 5 6)))(newline)
; CHECK-NEXT: (0 1 2 3 4 5 6)
(write (append '(0) nums '(4 5 6)))(newline)
; CHECK-NEXT: (0 1 2 3 4 5 6 1 2 3)
(write (append '(0) nums '(4 5 6) nums))(newline)
; CHECK-NEXT: (1 2 3)
(write nums)(newline)

;; List Formals
; CHECK-NEXT: #t
(write
  ((lambda args
     (null? args))))
(newline)

; CHECK-NEXT: (1 2 4 5 6)
(write
  ((lambda args
     args)
   1 2 4 5 6))
(newline)

; CHECK-NEXT: #(1 ())
(write
  ((lambda (arg . args)
     #(arg args))
   1))
(newline)

; CHECK-NEXT: #(1 2 ())
(write
  ((lambda (arg1 arg2 . args)
     #(arg1 arg2 ()))
   1 2))
(newline)

; CHECK-NEXT: #(1 (2))
(write
  ((lambda (arg . args)
     #(arg args))
   1 2))
(newline)

; CHECK-NEXT: (foo bar 4 5 6)
(write
  ((lambda (foo . bar)
     (append foo bar))
   '(foo bar) 4 5 6))
(newline)

; CHECK-NEXT: (foo bar 4 5 6)
(write
  ((lambda (foo . bar)
     (append foo bar))
   '(foo bar) 4 5 6))
(newline)

; CHECK-NEXT: (foo bar 4 5 6)
(write
  ((lambda (foo bar . baz)
     (append foo (list bar) baz))
   '(foo bar) 4 5 6))
(newline)

; CHECK-NEXT: (0 1 2 3 4)
(write
  (list (length '())
        (length '(9))
        (length '(9 42))
        (length '(9 (foo bar) 42))
        (length '((foo bar) 42 9 moo))))
