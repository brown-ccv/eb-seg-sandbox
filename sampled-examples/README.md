## Rose

Installation:

```bash
pipx install cylc-rose  # this will install cylc and rose
pipx install uv  # required for faster setup
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


## Looping through the case list



```bash
cylc stop sampled-examples/*;
cylc install . &&
cylc play sampled-examples \
--icp 2004-07-25 --fcp 2004-07-26 \
--set=BBOX="-812500.0,-2112500.0,-712500.0,-2012500.0" \
--set=LOCATION="'baffin_bay'" && # note that this string has to be "'double quoted'"
cylc tui sampled-examples
```


```bash
cylc stop sampled-examples/*;
cylc clean sampled-examples

datafile="all-cases.csv"
index_col="fullname"
for fullname in $(pipx run util/get_fullnames.py "${datafile}" "${index_col}"); 
do   
  cylc install . --run-name=${fullname}
  cylc play sampled-examples/${fullname} $(pipx run util/template.py ${datafile} ${index_col} ${fullname}); 
done

cylc tui
```