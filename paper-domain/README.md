## Cylc
To run the `cylc` workflow with the test data, run:
```bash
cylc stop ebseg-paper-domain*/*
cylc validate .

cylc install . -n ebseg-paper-domain-06
cylc play ebseg-paper-domain-06 --initial-cycle-point=2006-05-04 --final-cycle-point=2006-05-06

cylc install . -n ebseg-paper-domain-08
cylc play ebseg-paper-domain-08 --initial-cycle-point=2008-07-13 --final-cycle-point=2008-07-15

cylc tui
```

## OSCAR

The same Cylc configuration can be used on OSCAR, with the settings in `cylc/oscar/global.cylc`.
Install those using:
```bash
mkdir -p ~/.cylc/flow
cp ./oscar/global.cylc ~/.cylc/flow/global.cylc
```
