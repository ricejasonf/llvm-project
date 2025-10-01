; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

(define ovl
  (case-lambda
    (() 'zero)
    ((arg1) (list 'one arg1))
    ((arg1 arg2) (list 'two arg1 arg2))
    ((arg1 arg2 arg3) (list 'three arg1 arg2 arg3))
    (rest-args (list 'rest rest-args))))

; CHECK: zero
(write (ovl))
(newline)

; CHECK: (one 1)
(write (ovl 1))
(newline)

; CHECK: (two 1 2)
(write (ovl 1 2))
(newline)

; CHECK: (three 1 2 3)
(write (ovl 1 2 3))
(newline)

; CHECK: (rest (1 2 3 4))
(write (ovl 1 2 3 4))
(newline)

; CHECK: (rest (1 2 3 4 5))
(write (apply ovl '(1 2 3 4 5)))
(newline)

(define (ovl-capture x)
  (define y 'yo)
  (case-lambda
    (() (list 'zero x y))
    ((arg1) (list 'one x y arg1))
    ((arg1 arg2) (list 'two x y arg1 arg2))
    ((arg1 arg2 arg3) (list 'three x y arg1 arg2 arg3))
    (rest-args (list 'rest x y rest-args))))

(define ovl (ovl-capture 42))

; CHECK: (zero 42 yo)
(write (ovl))
(newline)

; CHECK: (one 42 yo 1)
(write (ovl 1))
(newline)

; CHECK: (two 42 yo 1 2)
(write (ovl 1 2))
(newline)

; CHECK: (three 42 yo 1 2 3)
(write (ovl 1 2 3))
(newline)

; CHECK: (rest 42 yo (1 2 3 4))
(write (ovl 1 2 3 4))
(newline)

; CHECK: (rest 42 yo (1 2 3 4 5))
(write (apply ovl '(1 2 3 4 5)))
(newline)
