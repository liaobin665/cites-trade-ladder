# Data

**No trade records are redistributed in this repository.** What is included are
aggregate results, figures and the term-to-ladder mapping. The raw tables can be
rebuilt from the official CITES download with the recipe below.

## 1. Official source

| Item | Where |
|---|---|
| CITES Trade Database (annual trade records) | https://trade.cites.org/ |
| Review of Significant Trade (selection lists and process) | https://cites.org/eng/imp/sigtradereview |
| Resolution Conf. 12.8 (Rev. CoP18), Review of Significant Trade | CITES document |
| Resolution Conf. 11.17 (Rev. CoP19), National reports | CITES document |
| Resolution Conf. 16.7 (Rev. CoP19), Non-detriment findings | CITES document |

Download the annual trade records for all species and all terms, 1975–2024, in
the "comparative tabulation" or raw-record format. Country names, taxon names
and term labels should be normalised to the spellings used by the database
before aggregation.

## 2. What the code expects

`data/links_l4.parquet` — one row per
`(Year, Exporter, Importer, Taxon, Term)`:

| Column | Type | Meaning |
|---|---|---|
| `Year` | int | Year of the record |
| `Exporter` | string | Exporter country/territory |
| `Importer` | string | Importer country/territory |
| `Taxon` | string | Species or higher taxon name |
| `Term` | string | CITES term (e.g. `live`, `skins`, `leather products (small)`) |
| `n_rec` | int | Number of underlying trade records in the cell |
| `n_appI` | float | Records in the cell whose specimen is Appendix I |
| `n_appII` | float | Records whose specimen is Appendix II |
| `n_srcW` | float | Records with source code `W` (wild) |
| `n_purT` | float | Records with purpose code `T` (commercial) |
| `n_repE` | float | Records reported by the exporter |
| `n_repI` | float | Records reported by the importer |

Aggregation recipe:

1. Keep records with `1975 <= Year <= 2024`.
2. Group by `Year, Exporter, Importer, Taxon, Term`.
3. `n_rec` = number of records in the group.
4. `n_appI`, `n_appII`, `n_srcW`, `n_purT`, `n_repE`, `n_repI` = number of
   records in the group with the corresponding attribute.
5. Save as parquet.

The published analysis has **29,158,995 records**, which collapse to
**2,456,696 aggregated rows**, covering **22,128 taxa**, **262 countries or
territories** and **115 term values**. If your rebuild is far from these
numbers, check the country-name normalisation first: spelling variants of the
same country split one unit into two and inflate the row count.

## 3. Term ladder

`config/term_ladder.csv` maps the 115 term values to four processing levels:

| Level | Meaning | Terms |
|---|---|---|
| S0 | live animals/plants and whole specimens | 9 |
| S1 | raw material and primary cuts | 49 |
| S2 | semi-processed | 25 |
| S3 | finished products | 31 |
| — | form not specified (e.g. `unspecified`) | 1 |

The mapping is an analytical construct, not a risk score. It was written by the
author and is deliberately shipped as a plain CSV so that readers can replace
it and re-run the pipeline. Columns: `term`, `ladder`, `confidence`, `rationale`
(the rationale text is in Chinese).

## 4. Terms of use

The CITES Trade Database is published by the CITES Secretariat. Check the terms
of use on the download page before redistributing any extract. This repository
therefore ships no raw records, no unit-level tables and no species-level case
files.
