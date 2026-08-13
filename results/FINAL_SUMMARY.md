# Final results

## 4b -- trained ON A-only (is the annotation signal learnable?)

- **bus_bra**: A-only = 0.5036 (sd 0.0205) vs chance 0.5000 -- 1.0x chance
- **fetal_planes**: A-only = 0.5945 (sd 0.0185) vs chance 0.1667 -- 3.6x chance

### bus_bra

```
              bal_acc      sd  chance     ece  n
test_variant                                    
a_full         0.5044  0.0201     0.5  0.1877  6
a_only         0.5036  0.0205     0.5  0.1083  6
a_removed      0.5016  0.0246     0.5  0.1831  6
```

### fetal_planes

```
              bal_acc      sd  chance     ece  n
test_variant                                    
a_full         0.2172  0.0502  0.1667  0.6127  6
a_only         0.5945  0.0185  0.1667  0.0404  6
a_removed      0.2034  0.0457  0.1667  0.6245  6
```

## 4c -- positive control (what does removing a USED shortcut cost?)

| strength | a_full | a_full__nomarker | drop | a_only | chance |
|---|---|---|---|---|---|
| 1.0 | 0.9993 | 0.6741 | **+0.3252** | 0.5784 | 0.1667 |
| 0.75 | 0.9715 | 0.9532 | **+0.0183** | 0.1819 | 0.1667 |
| 0.5 | 0.9612 | 0.9566 | **+0.0046** | 0.1805 | 0.1667 |
| 0.25 | 0.9542 | 0.9559 | **-0.0017** | 0.1624 | 0.1667 |

### How to read this

`drop` is what removing a shortcut the model demonstrably uses costs.
The real annotation-removal cost measured in the Step-4 gate was **-0.0002**
on fetal_planes, against a run-to-run SD of 0.0037.

- If `drop` at strength 1.0 is large, the A-removed cell CAN detect a
  shortcut at 0.7% pixel area, so the real null is a genuine null.
- If `drop` at strength 1.0 is also near zero, the design cannot detect any
  feature of this size, and that limitation is the finding -- not the null.
- The strength column gives the sensitivity floor: the weakest
  annotation-label correlation this design can see.

