; RUN: heavy-scheme --module-path=%heavy_module_path %s 2>&1 | FileCheck %s
(import (heavy base))

(define-syntax get
  (syntax-rules ()
    ((get key ...)
     (let ()
      (list (syntax-source-loc key) ...)))
    ))
(define five 'five)

(define GetList (get "here1"
                     "here2"
                     five
                     5))

; CHECK: source-loc.scm:12:21
; CHECK-NEXT: (define GetList (get "here1"
; CHECK: source-loc.scm:13:21
; CHECK-NEXT: "here2"
; CHECK: source-loc.scm:14:21
; CHECK-NEXT: five
; CHECK: source-loc.scm:15:21
; CHECK-NEXT: 5
(define NewList
  (do ((List GetList (cdr List)))
       (NewList '()
                (source-cons (car List) (cdr List)
                             (source-loc List)))
      ((null? List) NewList)
    ))

(do ((List NewList (cdr List)))
    ((null? List) 'done)
  (dump-source-loc (car List)))
