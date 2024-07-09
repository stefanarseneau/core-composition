## Core Compositions
---

The objective of this project is to measure the core compositions of white dwarfs. The catalog of this project can be built using the following command:
```
python -m corecomposition build [BASE CATALOG] [CONFIG FILE] [SAVE FILE]
```
and the analysis can be performed with the command
```
python -m corecomposition analyze [TARGET FILE] [CONFIG FILE] [SAVE FILE] --obspath=[OBSERVATION PATH]
```

---

**Gold Sample Threshold**
```
- max main sequence e_rv < 2.5
- wd and ms ruwe < 1.2
- wd and ms bp_rp_excess < 1.2
- all wd radius chi2s < 5
- all wd radius / error > 5
- average wd radius < 0.0075
```
no. targets: 5

**Silver Sample Threshold**
```
- max main sequence e_rv < 2.5
- wd and ms ruwe < 1.2
- wd and ms bp_rp_excess < 1.25
- all wd radius chi2s < 5
- all wd radius / error > 5
- average wd radius < 0.0075
```
no. targets: 14

**Bronze Sample Threshold**
```
- max main sequence e_rv < 2.5
- wd and ms ruwe --- NONE
- wd and ms bp_rp_excess --- NONE
- all wd radius chi2s < 5
- all wd radius / error > 5
- average wd radius < 0.0075
```
no. targets: 42

---

**Notes**
* 07/08/2024 #1. The hydrogen-rich models should have a larger radius than the hydrogen-poor models. This does not seem to be the case for the proof-of-concept data with a reliable gravz measurement. The bp-rp excess factor of this source is high, so it's possible that something (e.g. an unresolved brown dwarf) is messing with the measurement.
