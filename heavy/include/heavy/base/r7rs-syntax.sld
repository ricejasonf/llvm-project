(import (heavy builtins))

; Some of these are borrowed from the R7RS manual
; with changes to take advantage of heavy builtins.
(define-library (heavy base r7rs-syntax)
  (import (heavy builtins))
  (begin
    ; lambda already gives a nice error for an empty body.
    (define-syntax letrec*
      (syntax-rules ()
        ((letrec* ((var init) ...) body ...)
         ((lambda () (define var init) ...  body ...)))))

    (define-syntax letrec
      (syntax-rules ()
        ((letrec stuff) (letrec* stuff))))

    (define-syntax let
      (syntax-rules ()
        ((let ((name val) ...) body ...)
          ((lambda (name ...) body ...) val ...))
        ((let tag ((name val) ...) body ...)
          ((letrec ((tag (lambda (name ...)
            body ...)))
            tag)
            val ...))))

    (define-syntax cond
      (syntax-rules (else =>)
        ((cond (else result1 result2 ...))
         (begin result1 result2 ...))
        ((cond (test => result))
         (let ((temp test))
           (if temp (result temp))))
        ((cond (test => result) clause1 clause2 ...)
         (let ((temp test))
           (if temp
             (result temp)
             (cond clause1 clause2 ...))))
        ((cond (test)) test)
        ((cond (test) clause1 clause2 ...)
         (let ((temp test))
           (if temp
             temp
             (cond clause1 clause2 ...))))
        ((cond (test result1 result2 ...))
         (if test (begin result1 result2 ...)))
        ((cond (test result1 result2 ...)
               clause1 clause2 ...)
         (if test
           (begin result1 result2 ...)
           (cond clause1 clause2 ...)))))

    (define-syntax case
      (syntax-rules (else =>)
        ((case (key ...)
           clauses ...)
         (let ((atom-key (key ...)))
           (case atom-key clauses ...)))
        ((case key
           (else => result))
         (result key))
        ((case key
           (else result1 result2 ...))
         (begin result1 result2 ...))
        ((case key
           ((atoms ...) result1 result2 ...))
         (if (memv key '(atoms ...))
           (begin result1 result2 ...)))
        ((case key
           ((atoms ...) => result))
         (if (memv key '(atoms ...))
           (result key)))
        ((case key
           ((atoms ...) => result)
           clause clauses ...)
         (if (memv key '(atoms ...))
           (result key)
           (case key clause clauses ...)))
        ((case key
           ((atoms ...) result1 result2 ...)
           clause clauses ...)
         (if (memv key '(atoms ...))
           (begin result1 result2 ...)
           (case key clause clauses ...)))))

    (define-syntax and
      (syntax-rules ()
        ((and) #t)
        ((and test) test)
        ((and test1 test2 ...)
         (if test1 (and test2 ...) #f))))

    (define-syntax or
      (syntax-rules ()
        ((or) #f)
        ((or test) test)
        ((or test1 test2 ...)
         (let ((x test1))
           (if x x (or test2 ...))))))

    (define-syntax when
      (syntax-rules ()
        ((when test result1 result2 ...)
         (if test
           (begin result1 result2 ...)))))

    (define-syntax unless
      (syntax-rules ()
        ((unless test result1 result2 ...)
         (if (not test)
           (begin result1 result2 ...)))))

    (define-syntax guard
      (syntax-rules ()
        ((guard (var clause ...) e1 e2 ...)
         ((call/cc
            (lambda (guard-k)
              (with-exception-handler
                (lambda (condition)
                  ((call/cc
                     (lambda (handler-k)
                       (guard-k
                         (lambda ()
                           (define var condition)
                           (guard-aux
                             (handler-k
                               (lambda ()
                                 (raise-continuable condition)))
                             clause ...)))))))
                (lambda ()
                  (call-with-values
                    (lambda () e1 e2 ...)
                    (lambda args
                      (guard-k
                        (lambda ()
                          (apply values args)))))))))))))

    (define-syntax guard-aux
      (syntax-rules (else =>)
        ((guard-aux reraise (else result1 result2 ...))
         (begin result1 result2 ...))
        ((guard-aux reraise (test => result))
         (let ((temp test))
           (if temp
             (result temp)
             reraise)))
        ((guard-aux reraise (test => result)
                    clause1 clause2 ...)
         (let ((temp test))
           (if temp
             (result temp)
             (guard-aux reraise clause1 clause2 ...))))
        ((guard-aux reraise (test))
         (or test reraise))
        ((guard-aux reraise (test) clause1 clause2 ...)
         (let ((temp test))
           (if temp
             temp
             (guard-aux reraise clause1 clause2 ...))))
        ((guard-aux reraise (test result1 result2 ...))
         (if test
           (begin result1 result2 ...)
           reraise))
        ((guard-aux reraise
                    (test result1 result2 ...)
                    clause1 clause2 ...)
         (if test
           (begin result1 result2 ...)
           (guard-aux reraise clause1 clause2 ...)))))
    ) ; end begin
  (export
    let letrec letrec*
    cond case
    and or when unless
    ))
