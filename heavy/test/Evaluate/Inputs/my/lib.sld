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
        ((create-op-literal (EmptyThunk) X)
          ((lambda (Arg)
            ; Test hygiene of define initializer.
            (define (MakeAttr Z)
              (value-attr Z))
            ; Trigger compiling define initializers with a syntactic closure.
            (EmptyThunk)
            (create-op "heavy.literal"
              (attributes
                `("info", (MakeAttr Arg))))) X))))

    (define-syntax lam
      (syntax-rules ()
        ((lam body)
         ((lambda () body)))))

    (define-syntax my-lambda
      (syntax-rules (:)
        ((my-lambda ((arg : type) ...) body ...)
         (lambda (arg ...)
           body ...))))

    (write "end of init")
    (newline)
    ) ; end begin

  (export hello-module
          hello-module-syntax
          create-op-literal
          lam
          my-lambda)
)
(write "end of module")
(newline)
