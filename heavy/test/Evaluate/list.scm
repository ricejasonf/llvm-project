; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
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
(newline)

; CHECK-NEXT: ()
(write
  (make-list 0))
(newline)

; CHECK-NEXT: (0)
(write
  (make-list 1 0))
(newline)

; CHECK-NEXT: (0 0 0 0 0)
(write
  (make-list 5 0))
(newline)

; CHECK-NEXT: (foo bar baz)
(write
  (let ((List (make-list 3)))
    (list-set! List 0 'foo)
    (list-set! List 1 'bar)
    (list-set! List 2 'baz)
    List
    ))
(newline)

; CHECK-NEXT: (foo bar 0)
(write
  (let ((List (make-list 3 0)))
    (list-set! List 0 'foo)
    (list-set! List 1 'bar)
    List
    ))
(newline)

; CHECK-NEXT: (baz bar foo)
(write
  (let ((List (list 'foo 'bar 'baz)))
    (list (list-ref List 2)
          (list-ref List 1)
          (list-ref List 0))))
(newline)

; CHECK-NEXT: ()
(write
  (reverse '()))
(newline)

; CHECK-NEXT: (1)
(write
  (reverse '(1)))
(newline)

; CHECK-NEXT: (5 4 3 2 1)
(write
  (reverse '(1 2 3 4 5)))
(newline)

; CHECK-NEXT: (baz uhh)
(write
  (cadr '((foo bar) (baz uhh))))
(newline)

; CHECK-NEXT: (#t #f #t #f)
(write
  (map <
    '(0 1 1 3)
    '(1 2 2 7)
    '(2 0 3 3)))
(newline)

; CHECK-NEXT: ()
(write
  (map + '() '() '()))
(newline)

; CHECK-NEXT: (5)
(write
  (map + '(5)))
(newline)

; CHECK-NEXT: (7)
(write
  (map + '(5) '(2)))
(newline)

; CHECK-NEXT: (#(a b c) #(d e f) #(g h i))
(write
  (map vector
    '(a d g 'oof)
    '(b e h)
    '(c f i)))
(newline)

; CHECK-NEXT: (#(a b c) #(d e f) #(g h i))
(write
  (map vector
    '(a d g)
    '(b e h 'oof)
    '(c f i)))
(newline)

; CHECK-NEXT: (#(a b c) #(d e f) #(g h i))
(write
  (map vector
    '(a d g)
    '(b e h)
    '(c f i 'oof)))
(newline)
