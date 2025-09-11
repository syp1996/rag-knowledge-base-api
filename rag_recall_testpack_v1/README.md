# RAG Recall Test Pack v1

- 8 markdown docs + 1 price-table doc + queries.csv
- Designed to test: exact match, negation, paraphrase, acronym, bilingual, temporal, numeric, table lookup.

## How to use
1) Ingest all `*.md` files (and the table file if your pipeline supports CSV-like text).
2) Run the queries in `queries.csv` through your `/ask` endpoint.
3) Score a hit if the returned context/answer contains `expected_span` and the `target_doc` is among top-k (e.g., k<=5).

## Suggested metrics
- Top-1 / Top-3 recall
- MRR (mean reciprocal rank)
- Accuracy on each `type` (negation/temporal/etc.)

## Notes
- Each doc ends with `DOCID: ...` which you can surface in highlights to simplify scoring.
