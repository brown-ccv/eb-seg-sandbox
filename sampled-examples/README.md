## Rose

```bash
pipx install cylc-rose  # this will install cylc and rose
```

## Cylc
To run the `cylc` workflow with the test data, run:
```bash
cylc stop sampled-examples/*;
cylc validate . &&
cylc install . &&
cylc play sampled-examples &&
cylc tui sampled-examples 
```

## OSCAR

The same Cylc configuration can be used on OSCAR, with the settings in `cylc/oscar/global.cylc`.
Install those using:
```bash
mkdir -p ~/.cylc/flow
cp ./cylc/oscar/global.cylc ~/.cylc/flow/global.cylc
```



```

[template variables]
START="2006-05-04"
END="2006-05-06"
SATELLITE="aqua", "terra"
LOCATION="beaufort_sea"

```
