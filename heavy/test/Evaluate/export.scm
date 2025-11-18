; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy builtins))

;; Support early exports.
(define-library (testing export)
  (export some-fn-1)
  (export (rename some-fn-1 some-fn-2))
  (export (rename write my-write))
  (export (rename write my-write-2))
  (import (heavy builtins))
  (begin
    ;(define my-write 5) ;; would raise error
    (define (some-fn-1)
      (write "Hello from some-fn-1")
      (newline))
    )
  ;(export my-write) ;; would raise error
  )

(import (testing export))

; CHECK: "Hello from some-fn-1"
(some-fn-1)

; CHECK: "Hello from some-fn-1"
(some-fn-2)
