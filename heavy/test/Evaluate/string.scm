; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

; CHECK: (#t #t #f #f #f)
(write (list (string? "")
             (string? "foo")
             (string? 'moo)
             (string? 5)
             (string? ())))
(newline)

; CHECK-NEXT: (0 1 3 3 13 1 3 9)
(write (list (string-length "")
             (string-length "f")
             (string-length "foo")
             (string-length 'foo)
             (string-length "Hello, world!")
             (string-length "λ")
             (string-length "ΣλΣ")
             (string-length "ΣλΣfooΣλΣ")))
(newline)

; CHECK-NEXT: ""
(write (string-append))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-append "Hello, world!"))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-append "Hello" ", " "world!"))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-append 'Hello ", world" '!))(newline)

; CHECK-NEXT: "λΣλΣbarΣλΣfooΣλΣ"
(write (string-append "λ" "ΣλΣ" "bar" "ΣλΣfooΣλΣ"))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-copy "Hello, world!"))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-copy "Hello, world!" 0))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-copy "Hello, world!" 0 13))(newline)

; CHECK-NEXT: "Hello, world!"
(write (string-copy "Hello, world!" 0 42))(newline)

; CHECK-NEXT: "Hello"
(write (string-copy "Hello, world!" 0 5))(newline)

; CHECK-NEXT: "worl"
(write (string-copy "Hello, world!" 7 11))(newline)

; CHECK-NEXT: "!"
(write (string-copy "Hello, world!" 12 13))(newline)

; CHECK-NEXT: ""
(write (string-copy "Hello, world!" 12 12))(newline)

; CHECK-NEXT: ""
(write (string-copy "Hello, world!" 13 13))(newline)

; CHECK-NEXT: ""
(write (string-copy "Hello, world!" 13 5))(newline)

; CHECK-NEXT: #\Σ
(write (string-ref "ΣλΣfooΣλΣ" 0))(newline)

; CHECK-NEXT: #\λ
(write (string-ref "ΣλΣfooΣλΣ" 1))(newline)

; CHECK-NEXT: #\f
(write (string-ref "ΣλΣfooΣλΣ" 3))(newline)

; CHECK: "λ"
(write (string-copy "λ"))(newline)
; CHECK: "ΣλΣ"
(write (string-copy "ΣλΣ"))(newline)
; CHECK: "ΣλΣfooΣλΣ"
(write (string-copy "ΣλΣfooΣλΣ"))(newline)

; CHECK: ""
(write (string-copy "λ" 1))(newline)
; CHECK: "λΣ"
(write (string-copy "ΣλΣ" 1))(newline)
; CHECK: "fooΣ"
(write (string-copy "ΣλΣfooΣλΣ" 3 7))(newline)

; CHECK-NEXT: "0"
(write (number->string 0))(newline)

; CHECK-NEXT: "1"
(write (number->string 1))(newline)

; CHECK-NEXT: "42"
(write (number->string 42))(newline)

; CHECK-NEXT: "34567"
(write (number->string 34567))(newline)

; CHECK-NEXT: "3.1400001"
(write (number->string 3.14))(newline)

