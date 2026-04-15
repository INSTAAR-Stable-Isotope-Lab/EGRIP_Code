# EGRIP Crunch

`EGRIP Crunch` is software for processing ultra-high-resolution water stable isotope data from the East Greenland Ice-core Project (EGRIP).

This repository currently contains the main processing script and an interactive helper tool used during QA/QC flagging workflows.

## Repository Contents

- `EGRIP_Crunch_1.7b.py` - Primary processing workflow for isotope data.
- `pick_points.py` - Interactive point-picking utility for assigning manual flags from plots.
- `.zenodo.json` - Zenodo metadata for software archiving and citation.

## Version

Current software version: `1.7b`

## Requirements

The scripts import the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas` (used by `pick_points.py`)

Notes:

- `EGRIP_Crunch_1.7b.py` uses legacy Python 2 style syntax in places.
- Both scripts currently set the Matplotlib backend to `Qt4Agg`, which may require adjustment on modern systems.

## Project Metadata

- **Title:** EGRIP Crunch
- **Upload type:** software
- **Access:** open
- **Keywords:** ice-core science, water stable isotopes, EGRIP
- **Funding:** NSF award #1804098
- **Community:** data-processing-software

## People

### Creator

- Valerie A. Morris (INSTAAR, University of Colorado Boulder)

### Contributors

- Aaron J. Vimont (INSTAAR, University of Colorado Boulder)
- Vasileios Gkinis (Niels Bohr Institute, University of Copenhagen, Denmark)
- Tyler R. Jones (INSTAAR, University of Colorado Boulder)
- James W.C. White (INSTAAR, University of Colorado Boulder)
- Bruce H. Vaughn (INSTAAR, University of Colorado Boulder)

## License

GNU Lesser General Public License v3.0 or later

