; RUN: heavy-scheme --mode=read %s 2>&1 | FileCheck  %s --strict-whitespace --match-full-lines

#\a
; CHECK:#\a

#\A
; CHECK:#\A

#\b
; CHECK:#\b

#\x7 #\alarm
; CHECK:#\alarm
; CHECK:#\alarm

#\x08 #\backspace
; CHECK:#\backspace
; CHECK:#\backspace

#\x7F #\delete
; CHECK:#\delete
; CHECK:#\delete

#\x1b #\escape
; CHECK:#\escape
; CHECK:#\escape

#\xA #\newline
; CHECK:#\newline
; CHECK:#\newline

#\x00 #\x0 #\null
; CHECK-COUNT-3:#\null

#\x000d #\return
; CHECK-COUNT-2:#\return

#\  #\space
; CHECK-COUNT-2:#\space

#\x09  #\tab
; CHECK-COUNT-2:#\tab

#\λ
; CHECK:#\λ

#\☕
#\x2615
; CHECK-COUNT-2:#\☕
