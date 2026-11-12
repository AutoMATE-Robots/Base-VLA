Outputs of T6 go here:
  w.json    {"features": [...], "w": [...], "b": ...}   -> evidence/legibility.py
  sem.json  {"kind": "logistic", "a": ..., "b": ...}   -> evidence/relevance.SemanticCalibration
Until these exist the code runs with identity/placeholder values and warns UNCALIBRATED.
