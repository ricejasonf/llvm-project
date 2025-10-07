(import (heavy builtins))

(define-library (heavy base list)
  (import (heavy builtins)
          (heavy base r7rs-syntax))
  (begin
    (define (caar x) (car (car x)))
    (define (cadr x) (car (cdr x)))
    (define (cdar x) (cdr (car x)))
    (define (cddr x) (cdr (cdr x)))

    (define (member-fast obj list compare)
      (if (pair? list)
        (if (compare obj (car list))
          list
          (member-fast obj (cdr list) compare))
        #f))
    (define member
      (case-lambda
        ((member obj list compare)
            (member-fast obj list compare))
        ((member obj list)
            member obj list equal?)))
    (define (memq obj list)
      (member-fast obj list eq?))
    (define (memv obj list)
      (member-fast obj list eqv?))

    (define (reverse InputList)
      (let Loop ((List InputList)
                 (NewList '()))
        (if (pair? List)
          (Loop (cdr List)
            (cons (car List) NewList))
          NewList)))

    ) ; end of begin
  (export
    caar cadr cdar cddr
    member memq memv
    reverse
    )
)
