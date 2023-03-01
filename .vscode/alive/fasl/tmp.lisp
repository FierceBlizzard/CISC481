(defvar *puzzle-0*'((3 1 2)
                       (7 nil 5)
                       (4 6 8)))

(defun possible-actions (puzzle)
    (dotimes (row 3)
        (dotimes (col 3)
            (when 
                (null (aref puzzle))
            )
        )
    )
    
)

(defun result (puzzle)
    (write (position 1 puzzle))
)
(result *puzzle-0*)