; RUN: heavy-scheme %s | FileCheck %s

(import (heavy builtins))

; CHECK: 17
(write (+ (+ (+ 2 3) (+ 5 7))))
(newline)
