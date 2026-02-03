; RUN: heavy-scheme --module-path=%heavy_module_path %s | FileCheck %s
(import (heavy builtins))

;; Support early exports.
(define-library (testing export)
  (export some-fn-1
          (rename some-fn-1 some-fn-2)
          (rename write my-write)
          (rename write my-write-2)
          some-syntax
          (rename some-syntax some-syntax-2))
  (import (heavy builtins))
  (begin
    ;(define my-write 5) ;; would raise error
    (define (some-fn-1)
      (write "Hello from some-fn-1")
      (newline))
    (define-syntax some-syntax
      (syntax-rules ()
        ((some-syntax x) (begin (write x)
                                (newline)))))

    )
  ;(export my-write) ;; would raise error
  )

(import (testing export))

; CHECK: "Hello from some-fn-1"
(some-fn-1)

; CHECK: "Hello from some-fn-1"
(some-fn-2)

; CHECK: "Hello some-syntax"
(some-syntax "Hello some-syntax")

; CHECK: "Hello some-syntax... 2"
(some-syntax-2 "Hello some-syntax... 2")
