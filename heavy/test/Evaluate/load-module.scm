; RUN: heavy-scheme --module-path=%S/Inputs %s | FileCheck %s
; RUN: heavy-scheme --module-path=%S/../Evaluate/Inputs %s | FileCheck %s

; CHECK: "end of module"
; CHECK-NEXT: "end of init"
(import (my lib))
(import (only (heavy builtins)
              write newline))

; CHECK: "hello module!"
; CHECK-NEXT: 5
(hello-module 5)

; CHECK: "syntax: ""hello module!"
; CHECK-NEXT: woof
(hello-module-syntax "woof!")

; COM-CHECK: ???
(write (create-op-literal 42))
(newline)
