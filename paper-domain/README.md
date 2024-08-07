## Cylc
To run the `cylc` workflow with the test data, run:
```bash
cylc stop paper-domain/*
cylc validate .
cylc install . -n paper-domain
cylc play paper-domain
cylc tui paper-domain
```

## OSCAR

The same Cylc configuration can be used on OSCAR, with the settings in `cylc/oscar/global.cylc`.
Install those using:
```bash
mkdir -p ~/.cylc/flow
cp ./cylc/oscar/global.cylc ~/.cylc/flow/global.cylc
```
