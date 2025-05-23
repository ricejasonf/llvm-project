; RUN: heavy-scheme --module-path=./Inputs %s | FileCheck %s
; RUN: heavy-scheme --module-path=../Evaluate/Inputs %s | FileCheck %s

; CHECK: "end of module"
; CHECK-NEXT: "end of init"
(import (my lib))

; CHECK: "hello module!"
; CHECK-NEXT: 5
(hello-module 5)
