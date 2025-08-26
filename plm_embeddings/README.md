# Protein Language Model Embeddings

This part of the repository provides a unified interface for generating embeddings from multiple **protein language models (PLMs)**.  
The code integrates functionality from the original model repositories (see [Ankh](https://github.com/agemagician/Ankh), [ProtT5](https://github.com/agemagician/ProtTrans), [ProstT5](https://github.com/mheinzinger/ProstT5), [ESM1 and ESM2](https://github.com/facebookresearch/esm), as well as [ESM3 and ESMC](https://github.com/evolutionaryscale/esm)) and allows you to flexibly generate **per-residue embeddings** or **aggregated per-protein embeddings** with optional sequence chunking for long proteins. If you use this repository to generate embeddings, please cite the **original model papers** corresponding to the models you used.

All implemented models are listed in [`plm_embeddings/embedding_model_metadata.json`](plm_embeddings/embedding_model_metadata.json).

---

## Features

- Supports multiple PLMs (see metadata file for available models).
- Option to extract embeddings:
  - **Per residue** (embedding for each amino acid).
  - **Per protein** (aggregated over residues).
- Aggregation options for long sequences:
  - Sequences longer than a given threshold can be **chopped into overlapping chunks and aggregated**.
  - Configurable overlap size and bidirectional chopping.
- Easy configuration via command-line arguments.

---


## Requirements

### Environments
Two separate environments are provided under `plm_embeddings/requirements/`:
- `emmaplm.yml` → for **ESM1, ESM2, Ankh, ProtT5, ProstT5**  
- `esm3.yml` → for **ESM3 models**  

Create and activate with:

```bash
conda env create -f plm_embeddings/requirements/esm3.yml
conda activate esm3

conda env create -f plm_embeddings/requirements/emmaplm.yml
conda activate emmaplm
````

###  Hugging Face Access

You need access to the pretrained models on Hugging Face.
- Create a Hugging Face account.
- Generate a user access token: [How to get a token](https://huggingface.co/docs/hub/en/security-tokens).

---

## Usage
The main script is `plm_embeddings/get_embeddings.py`.

Example:

```
python plm_embeddings/get_embeddings.py \
    --input data/proteins.fasta \
    --model esm3-sm-open-v1 \
    --output_dir outputs/esm3_embeddings \
    --per_protein True
```

### Arguments

|Argument |	Type |	Default |	Description |
| --- | --- | --- | ---  |
|--input, -i	 |str	 |required	 |Path to a FASTA file containing protein sequences or  a list of protein names|
|--model, -m	|str	|esm3-sm-open-v1	|Name of the embedding model to be used (see metadata file)|
|--output_dir, -o	|str	|required	|Output directory for storing embeddings|
|--run_id	|str	|None	|Unique identifier for the current run. If not provided, a new run ID will be generated using the current timestamp|
|--no_gpu	|flag	|False	|Flag to disable GPU usage|
|--dev	|flag	|False	|Flag to enable development mode (shortening input data)|
|--layer	|int	|-1	|Layer index to extract embeddings from|
|--per_protein	|bool	|False	|Aggregate embeddings per protein|
|--output_format	|str	|npy	|File format for the embeddings|
|--max_seq_length	|int	|1022	|Maximum sequence length. Sequences longer than this will be chopped.|
|--chunk_overlap	|int	|300	|Overlap size between chunks|
|--bidirectional	|flag	|True	|Flag to chop sequences from both directions|
|--model_dir |	str	| None | Directory for model output. Should be a non-empty string or missing.|