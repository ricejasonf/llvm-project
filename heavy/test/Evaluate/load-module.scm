; RUN: heavy-scheme --module-path=%heavy_module_path --module-path=%S/Inputs %s | FileCheck %s
; RUN: heavy-scheme --module-path=%heavy_module_path --module-path=%S/../Evaluate/Inputs %s | FileCheck %s

; CHECK: "end of module"
; CHECK-NEXT: "end of init"
(import (my lib))
(import (only (heavy base)
              write newline lambda
              + define
              ))

; CHECK: "hello module!"
; CHECK-NEXT: 5
(hello-module 5)

; CHECK: "syntax: ""hello module!"
; CHECK-NEXT: woof
(hello-module-syntax "woof!")

; CHECK: #op{"heavy.literal"() {info = #heavy<"42">}
(write (create-op-literal ((lambda () #f)) 42))
(newline)

; CHECK: 42
; CHECK-NEXT: 43
; CHECK-NEXT: 44
((my-lambda ((x : Int) (y : Int))
  (define X (+ x 1)) ; define in syntax closure
  (write x)
  (newline)
  (write X)
  (newline)
  (write y)) 42 44)
(newline)

; CHECK: 9000
(lam (lam (write 9000)))
(newline)
