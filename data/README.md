# Data Directory

Place these three files here before running anything.

## Required Files

### `expression_ef.csv`
RNA-seq raw counts from excitatory frontal neurons (cell-type sorted).

| Column       | Description                                    |
|--------------|------------------------------------------------|
| animal_id    | Unique animal identifier                       |
| timepoint    | One of: E15, P0, P13                           |
| condition    | One of: saline, polyIC                         |
| Gene1...GeneN| Raw count values (integers), one column per gene|

### `expression_wc.csv`
RNA-seq raw counts from whole cortex bulk tissue.

| Column       | Description                                    |
|--------------|------------------------------------------------|
| animal_id    | Unique animal identifier                       |
| timepoint    | One of: E15, P0, P70, P189                     |
| condition    | One of: saline, polyIC                         |
| Gene1...GeneN| Raw count values (integers), one column per gene|

Gene columns must match `expression_ef.csv` exactly (same names, same order).

### `metadata.csv`
One row per animal-region entry.

| Column     | Description                                      |
|------------|--------------------------------------------------|
| animal_id  | Unique animal identifier                         |
| timepoint  | One of: E15, P0, P13, P70, P189                  |
| condition  | One of: saline, polyIC                           |
| region     | One of: excitatory_frontal, whole_cortex         |

Same animal may appear twice (once per region) if both tissues were sequenced.

## Normalization

Raw counts are expected. The precompute scripts handle:
1. Library-size normalization (CPM, scaling to 1e6)
2. log1p transformation
