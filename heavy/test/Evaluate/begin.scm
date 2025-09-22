; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

; Test top level begin.
; CHECK: foo
; CHECK-NEXT: bar
(begin
  (write 'foo)
  (newline)
  (write 'bar)
  (newline))
