# Use FinSage

## Quick Summary

- **Purpose**: Route library users and agents to current public operations, inputs, results, examples, and guides.
- **Read when**: Calling FinSage as a library or selecting a supported public operation.
- **Load next**: Start with the component row that matches the requested operation; do not load unrelated component designs.
- **Import rule**: Import public callables from the owning package shown in the component catalog; ETL scaffolds come from `finsage.etl`, flow operations from `finsage.flows`, inputs and configs from `finsage.configs`, and result types from their canonical component package.

## Contents

- [Public Usage Sources](#public-usage-sources)
- [Where to Import From](#where-to-import-from)

## Public Usage Sources

Use this index when calling FinSage as a library. These links point to the current public API, focused examples, and component-specific guidance.

| Need | Read |
|---|---|
| Current operations, inputs, results, and sources | [Public component catalog](../../docs/README.md) |
| Installation and common usage | [Root README usage](../../README.md#usage-examples) |
| Source-first paper flow | [Paper flow design](../design/flow/paper.md) |
| News collection | [News design and behavior](../design/flow/news.md) |
| Search local knowledge by meaning | [Library guide](../../docs/library.md) and [focused example](../../examples/library/README.md) |
| Observable ETL scaffolds | [ETL guide](../../docs/etl.md) and [focused examples](../../examples/etl/) |
| Runnable operation examples | [`examples/flows/`](../../examples/flows/) |
| Focused preprocessing examples | [`examples/preprocess/`](../../examples/preprocess/) |

## Where to Import From

Use the public component catalog as the import authority for each capability. Observable ETL scaffolds are exported from `finsage.etl`, public flow operations and builders from `finsage.flows`, public inputs and configs from `finsage.configs`, cognitive services from `finsage.mind`, and result contracts from the canonical layer named in the catalog.
