(format t "Hello, World")
(terpri)
(defun avg (num1 num2)
    (+ num1 num2)
)
(format t "~d" (avg 10 10))
(terpri)
(defun how-hot (temp)
    (cond
        ((not (numberp temp)) (error))
        ((< temp 0) "really-cold")
        ((< temp 40) "cold")
        ((< temp 60) "cool")
        ((< temp 80) "nice")
        ((< temp 100) "hot")
        (t "really-hot")
    )
)
(format t (how-hot 45))
(terpri)

(defun checker (a b)
    (eq a b)
)
