; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

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
