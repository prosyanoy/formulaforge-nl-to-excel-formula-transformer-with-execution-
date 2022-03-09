# FormulaForge error analysis (test, beams=4)

Examples: **1000** | Exact match: **69.5%** | Execution accuracy: **77.8%** | Valid-formula rate: **99.8%**

83 predictions were execution-correct but not string-identical to gold (equivalent formulas that exact match would wrongly penalise).

## By difficulty tier

| Difficulty tier | N | Exact match | Execution acc. | Valid |
|---|---|---|---|---|
| 1 | 264 | 88.6% | 89.8% | 100.0% |
| 2 | 382 | 66.5% | 73.6% | 100.0% |
| 3 | 354 | 58.5% | 73.4% | 99.4% |

## By function family

| Function family | N | Exact match | Execution acc. | Valid |
|---|---|---|---|---|
| AVERAGE | 108 | 80.6% | 80.6% | 98.1% |
| COUNTIF | 124 | 92.7% | 92.7% | 100.0% |
| IF | 171 | 80.7% | 87.1% | 100.0% |
| INDEX_MATCH | 103 | 55.3% | 67.0% | 100.0% |
| MINMAX | 101 | 84.2% | 84.2% | 100.0% |
| SUM | 129 | 69.0% | 82.2% | 100.0% |
| SUMIFS | 171 | 40.4% | 57.9% | 100.0% |
| VLOOKUP | 93 | 59.1% | 73.1% | 100.0% |

## Execution accuracy: function family x tier

| Family | tier 1 | tier 2 | tier 3 |
|---|---|---|---|
| AVERAGE | 87.9% (n=58) | 76.2% (n=42) | 50.0% (n=8) |
| COUNTIF | 100.0% (n=54) | 86.1% (n=36) | 88.2% (n=34) |
| IF | 92.0% (n=50) | 79.7% (n=64) | 91.2% (n=57) |
| INDEX_MATCH | - | 43.4% (n=53) | 92.0% (n=50) |
| MINMAX | 82.0% (n=50) | 91.3% (n=23) | 82.1% (n=28) |
| SUM | 86.5% (n=52) | 82.1% (n=28) | 77.6% (n=49) |
| SUMIFS | - | 84.7% (n=72) | 38.4% (n=99) |
| VLOOKUP | - | 60.9% (n=64) | 100.0% (n=29) |

## Failure modes (execution mismatches)

| Failure mode | Count | Share of failures |
|---|---|---|
| wrong_reference | 189 | 85.1% |
| wrong_operator_or_number | 27 | 12.2% |
| wrong_function | 4 | 1.8% |
| invalid_formula | 2 | 0.9% |

## Example failures

### AVERAGE

- **invalid_formula** (tier 3): _Average Cost for which Grade is B and Discount is lower than 15 in Excel_
  - gold: `AVERAGEIFS(C2:C6,A2:A6,"B",B2:B6,"<15")` -> 310.0
  - pred: `AVERAGEIFS(B2:B6,A2:A6,"B",C2:C6,"<15")` -> '#ERROR'
- **wrong_reference** (tier 2): _Help me average Hours for North region_
  - gold: `AVERAGEIF(C2:C9,"North",B2:B9)` -> 26.666666666666668
  - pred: `AVERAGEIF(C2:C9,"North",D2:D9)` -> 37.0
- **wrong_reference** (tier 1): _Mean Revenue please_
  - gold: `AVERAGE(A2:A9)` -> 523.0
  - pred: `AVERAGE(C2:C9)` -> 317.375

### COUNTIF

- **wrong_reference** (tier 3): _Number of Open Status entries with Hours more than 30_
  - gold: `COUNTIFS(B2:B12,"Open",C2:C12,">30")` -> 1.0
  - pred: `COUNTIFS(D2:D12,"Open",C2:C12,">30")` -> 0.0
- **wrong_reference** (tier 2): _How many rows have Hours under 30?_
  - gold: `COUNTIF(B2:B6,"<30")` -> 4.0
  - pred: `COUNTIF(C2:C6,"<30")` -> 0.0
- **wrong_reference** (tier 2): _How many records have discount no less than 15?_
  - gold: `COUNTIF(A2:A15,">=15")` -> 3.0
  - pred: `COUNTIF(B2:B15,">=15")` -> 9.0

### IF

- **wrong_reference** (tier 2): _If Region in row 6 is North, increase Hours by 5 percent, otherwise keep Hours please_
  - gold: `IF(A6="North",B6*1.05,B6)` -> 42.0
  - pred: `IF(A6="North",C6*1.05,C6)` -> 88.2
- **wrong_reference** (tier 1): _Please check whether Age in row 3 is above 35; show Yes or No in the sheet_
  - gold: `IF(C3>35,"Yes","No")` -> 'No'
  - pred: `IF(A3>35,"Yes","No")` -> 'Yes'
- **wrong_reference** (tier 2): _I need to show "Target met" when the sum of Quantity is over 300, else "Below target"_
  - gold: `IF(SUM(A2:A13)>300,"Target met","Below target")` -> 'Below target'
  - pred: `IF(SUM(B2:B13)>300,"Target met","Below target")` -> 'Target met'

### INDEX_MATCH

- **wrong_reference** (tier 2): _Look up Grace in the Employee column and return the matching cost please_
  - gold: `INDEX(B2:B11,MATCH("Grace",C2:C11,0))` -> 11.0
  - pred: `INDEX(D2:D11,MATCH("Grace",C2:C11,0))` -> 'Red'
- **wrong_reference** (tier 2): _Return Grade where Employee is Tina in Excel_
  - gold: `INDEX(E2:E6,MATCH("Tina",D2:D6,0))` -> 'C'
  - pred: `INDEX(A2:A6,MATCH("Tina",D2:D6,0))` -> 79.0
- **wrong_reference** (tier 2): _Can you return Discount where Employee is Tina in the sheet?_
  - gold: `INDEX(B2:B8,MATCH("Tina",D2:D8,0))` -> 12.0
  - pred: `INDEX(C2:C8,MATCH("Tina",D2:D8,0))` -> 94.0

### MINMAX

- **wrong_reference** (tier 1): _Largest value in Price in Excel_
  - gold: `MAX(A2:A6)` -> 193.0
  - pred: `MAX(C2:C6)` -> 65.0
- **wrong_reference** (tier 3): _Smallest Quantity among records with Department equal to HR please_
  - gold: `MINIFS(B2:B15,A2:A15,"HR")` -> 1.0
  - pred: `MINIFS(D2:D15,A2:A15,"HR")` -> 5.0
- **wrong_reference** (tier 1): _Find the maximum Discount in Excel_
  - gold: `MAX(A2:A6)` -> 22.0
  - pred: `MAX(B2:B6)` -> 30.0

### SUM

- **wrong_reference** (tier 1): _Can you add up all the Revenue in the sheet?_
  - gold: `SUM(A2:A12)` -> 6650.0
  - pred: `SUM(C2:C12)` -> 962.0
- **wrong_reference** (tier 3): _Multiply Revenue by Price row by row and add it all up please_
  - gold: `SUMPRODUCT(C2:C15,B2:B15)` -> 778931.0
  - pred: `SUMPRODUCT(B2:B15,D2:D15)` -> 4656.8
- **wrong_reference** (tier 2): _Can you calculate the total Rating between rows 3 and 4 please?_
  - gold: `SUM(A3:A4)` -> 8.4
  - pred: `SUM(B3:B4)` -> 773.0

### SUMIFS

- **wrong_operator_or_number** (tier 3): _What is the total Revenue for A Grade and Apples Product in the sheet?_
  - gold: `SUMIFS(C2:C15,A2:A15,"A",B2:B15,"Apples")` -> 391.0
  - pred: `SUMIFS(C2:C15,B2:B15,"A",A2:A15,"Apples")` -> 0.0
- **wrong_function** (tier 2): _How do I sum the Rating for all rows with channel equal to Partner please?_
  - gold: `SUMIFS(D2:D6,B2:B6,"Partner")` -> 9.3
  - pred: `SUMIF(B2:B6,"Partner",A2:A6)` -> 33.0
- **wrong_reference** (tier 3): _Can you total Quantity where Department is Sales and Discount is less than or equal to 20?_
  - gold: `SUMIFS(C2:C14,A2:A14,"Sales",D2:D14,"<=20")` -> 99.0
  - pred: `SUMIFS(B2:B14,A2:A14,"Sales",D2:D14,"<=20")` -> 1351.0

### VLOOKUP

- **wrong_reference** (tier 2): _Look up Aperture and return its discount_
  - gold: `VLOOKUP("Aperture",A2:C6,3,FALSE)` -> 6.0
  - pred: `VLOOKUP("Aperture",A2:B6,2,FALSE)` -> 'South'
- **wrong_reference** (tier 2): _Use VLOOKUP to get channel for Gringotts from the table_
  - gold: `VLOOKUP("Gringotts",A2:E6,5,FALSE)` -> 'Partner'
  - pred: `VLOOKUP("Gringotts",A2:C6,3,FALSE)` -> 10.0
- **wrong_reference** (tier 2): _Use VLOOKUP to work out Discount for Quinn from the table_
  - gold: `VLOOKUP("Quinn",A2:E13,5,FALSE)` -> 4.0
  - pred: `VLOOKUP("Quinn",A2:C13,3,FALSE)` -> 13.0
