(import (heavy builtins))

(define-library (heavy base)
  (import (heavy builtins))
  (begin
    (define (caar x) (car (car x)))
    (define (cadr x) (car (cdr x)))
    (define (cdar x) (cdr (car x)))
    (define (cddr x) (cdr (cdr x)))

    ; FIXME Procedural macros because we do not keep Env alive.
    #;(define-syntax ir-macro-transformer
      (syntax-fn
        (lambda (Syntax Env)
          (define ProcSyntax (cadr (cadr Syntax)))
          ; Eval ProcSyntax to create Proc
          ; FIXME This requires keeping Env alive which
          ;       we don't do to support incremental compiling
          ;       of self contained functions.
          ;       This is why static templates such as with
          ;       syntax-rules are needed.
          ;       ie Procedural macros are right out.
          (define Proc (make-syntactic-closure Env '() ProcSyntax))
          (dump Proc)
          (dump Proc)
          (make-syntax-fn
            (lambda (syntax env)
              (define expr (cadr (inject syntax)))
              (define (inject x)
                (make-syntactic-closure env '() x))
              (define compare equal?)
              (define TemplateInst (Proc expr inject compare))
              (opgen TemplateInst env)))
        )))

    ) ; end of begin
  (export
    ; syntax
    define
    define-syntax
    if
    lambda
    quasiquote
    quote
    set!
    syntax-rules
    ir-macro-transformer
    begin
    cond-expand
    define-library
    export
    include
    include-ci
    include-library-declarations
    source-loc
    parse-source-file
    +
    -
    /
    *
    >
    <
    apply
    append
    call-with-values
    call/cc
    car
    cdr
    cons
    dump
    dynamic-wind
    eq?
    equal?
    eqv?
    error
    length
    list
    newline
    number->string
    raise
    string-append
    string-copy
    string-length
    values
    with-exception-handler
    write

    compile
    eval
    module-path
    op-eval

    ; Type predicates
    boolean?
    bytevector?
    char?
    eof-object?
    null?
    number?
    pair?
    port?
    procedure?
    string?
    symbol?
    vector?

    ; Extended types.
    mlir-operation?
    source-value?
    )
)
