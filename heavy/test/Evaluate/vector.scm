; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy base))

(define (id x) x)

; CHECK: #()
(write #())(newline)

; CHECK: #(5)
(write #(5))(newline)

; CHECK: #(42)
(write #((id 42)))(newline)

; CHECK: #(1 2 "yo" "moo")
(write #(1 2 "yo" (id "moo")))(newline)

; CHECK: #(#(1 2 "yo" "moo"))
(write #(#(1 2 "yo" (id "moo"))))(newline)

; CHECK: #()
(write (vector))(newline)

; CHECK: #(5)
(write (vector 5))(newline)

; CHECK: #(42)
(write (vector (id 42)))(newline)

; CHECK: #(1 2 "yo" "moo")
(write (vector 1 2 "yo" (id "moo")))(newline)

; CHECK: #(#(1 2 "yo" "moo"))
(write (vector (vector 1 2 "yo" (id "moo"))))(newline)

; CHECK-NEXT: #()
(write
  (make-vector 0))
(newline)

; CHECK-NEXT: #(0)
(write
  (make-vector 1 0))
(newline)

; CHECK-NEXT: #(0 0 0 0 0)
(write
  (make-vector 5 0))
(newline)

; CHECK-NEXT: #(foo bar baz)
(write
  (let ((Vec (make-vector 3)))
    (vector-set! Vec 0 'foo)
    (vector-set! Vec 1 'bar)
    (vector-set! Vec 2 'baz)
    Vec
    ))
(newline)

; CHECK-NEXT: #(foo bar 0)
(write
  (let ((Vec (make-vector 3 0)))
    (vector-set! Vec 0 'foo)
    (vector-set! Vec 1 'bar)
    Vec
    ))
(newline)

;CHECK-NEXT: #(0 1 2 3 4)
(write
  (vector (vector-length #())
          (vector-length #(1))
          (vector-length #(1 2))
          (vector-length #(1 2 3))
          (vector-length #(1 2 3 4))
          ))
(newline)

;CHECK-NEXT: (baz bar foo)
(write
  (let ((Vec (vector 'foo 'bar 'baz)))
    (list (vector-ref Vec 2)
          (vector-ref Vec 1)
          (vector-ref Vec 0))))
(newline)
