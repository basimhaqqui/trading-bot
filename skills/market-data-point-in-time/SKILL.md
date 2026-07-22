---
name: market-data-point-in-time
description: Build and audit replay-safe market datasets with separate event and availability timestamps, immutable raw events, instrument history, provenance, and data-quality checks. Use when adding a market-data adapter, ingesting trades/quotes/books/funding/on-chain/news/rules, constructing a historical universe, or investigating look-ahead and survivorship bias.
---

# Point-in-Time Market Data

Build data that reproduces exactly what was knowable at a decision time.

## Workflow

1. Read `src/trading_bot/core/schemas.py` and `src/trading_bot/core/store.py` before changing ingestion.
2. Define the instrument in the canonical master, including asset class, venue, multiplier, active interval, expiry, settlement, and source metadata.
3. Map each source record to `MarketEvent` with:
   - `event_time`: when the market or underlying event occurred.
   - `available_at`: earliest time the strategy could have received it.
   - Stable source event ID, venue sequence, source, and canonical instrument ID.
4. Preserve raw payloads. Add normalized fields without silently correcting history.
5. Reject conflicting duplicate IDs, negative sequences, naive datetimes, and availability before event time.
6. Test the adapter with delayed, corrected, duplicate, missing, and out-of-order records.
7. Query replay data only through `events_available_at(as_of)` or a stricter derived interface.

## Required checks

- Retain delisted/expired instruments and prior symbol mappings.
- Timestamp revisions and corporate actions when published, not by reporting period.
- Detect stale or crossed quotes, sequence gaps, resets, clock drift, and source outages.
- Preserve venue, feed, license/provenance, correction state, and ingestion time.
- Use executable quotes/books for fill research; never assume last trade or candle close was fillable.

Do not add broker credentials or order submission to a data adapter.
