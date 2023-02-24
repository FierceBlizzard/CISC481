(defvar *puzzle-0* #2A((3 1 2)
                       (7 nil 5)
                       (4 6 8)))

(defun possible-actions (puzzle)
    (declare (ignore puzzle))
    (list :left :right :up :down :row :col))