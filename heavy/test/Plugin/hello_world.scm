; RUN: heavy-scheme %s | FileCheck %s
(import (heavy builtins))

(load-plugin "libheavyHelloWorld.so")
(define my-write
  (load-builtin "heavy_hello_world_my_write"))

(define compute-answer
  (load-builtin "heavy_hello_world_compute_answer"))

; CHECK: (hello world)
(my-write '(hello world))


; CHECK: 42
(write (compute-answer))
(newline)

