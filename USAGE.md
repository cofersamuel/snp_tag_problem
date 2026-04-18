Usage (CLI)

Run the pipeline from the command line. Example:

```bash
conda activate cbio
python tfg_tagsnp_pymoo.py --mode fast --data-source synthetic
```

Options:
- `--mode` / `-m`: execution mode, one of `fast`, `medium`, `full`.
- `--data-source` / `-d`: data source, `synthetic` or `hinds2005`.

Note: `NUM_BLOCKS` remains a code-level setting inside `setup_configuration()`.
