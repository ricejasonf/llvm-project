(import (heavy builtins))

(define-library (my lib)
  (import (heavy builtins)
          (heavy mlir))
  (begin
    (define (hello-module x)
      (write "hello module!")
      (newline)
      (write x)
      (newline)
      )

    (define-syntax hello-module-syntax
      (syntax-rules ()
        ((hello-module-syntax x)
          (begin
            (write "syntax: ")
            (hello-module x))
          )))

    (define-syntax create-op-literal
      (syntax-rules ()
        ((create-op-literal (Arg) X)
          ((lambda (Arg)
            (create-op "heavy.literal"
              (attributes
                `("info", (value-attr Arg))))) X))))

    (write "end of init")
    (newline)
    ) ; end begin

  (export hello-module
          hello-module-syntax
          create-op-literal)
)
(write "end of module")
(newline)
