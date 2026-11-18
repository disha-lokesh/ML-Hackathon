Place the 10 reserved evaluator trips here on the API laptop, one folder per trip:
eval_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
The .route files are read only server-side for scoring. If this folder is empty
the API falls back to the student_test partition (development mode).
