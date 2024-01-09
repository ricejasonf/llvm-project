; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines

; CHECK:#\a
#\a

; CHECK:#\A
#\A

; CHECK:#\b
#\b

; CHECK:#\alarm
; CHECK:#\alarm
#\x7 #\alarm

; CHECK:#\backspace
; CHECK:#\backspace
#\x08 #\backspace

; CHECK:#\delete
; CHECK:#\delete
#\x7F #\delete

; CHECK:#\escape
; CHECK:#\escape
#\x1b #\escape

; CHECK:#\newline
; CHECK:#\newline
#\xA #\newline

; CHECK:#\null
; CHECK:#\null
; CHECK:#\null
#\x00 #\x0 #\null

; CHECK:#\return
; CHECK:#\return
#\x000d #\return

; CHECK:#\space
; CHECK:#\space
#\  #\space

; CHECK:#\tab
; CHECK:#\tab
#\x09  #\tab

; CHECK:#\λ
#\λ

; CHECK:#\☕
; CHECK:#\☕
#\☕
#\x2615
