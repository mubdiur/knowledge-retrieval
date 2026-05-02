# Payment Gateway — Runbook

## Service Overview
Payment gateway handles all payment processing, fraud detection, and refund operations.
**Owner:** Alice Wang (#payments-squad)
**PagerDuty:** PD-payments-prod

## Critical Contacts
- **Engineering:** @alice, @bob
- **SRE:** @frank
- **On-call:** Check PagerDuty schedule

## Common Failure Modes

### 1. Database Connection Exhaustion
**Symptoms:** Error rate spikes, "connection refused" in logs, PagerDuty alerts
**Detection:** Datadog monitor `payment.connections.used > 45`
**Triage:**
1. Check `pgbouncer` pool status: `show pools;`
2. Increase `max_connections` if needed
3. Look for slow queries: `pg_stat_activity`
4. Kill idle-in-transaction connections
**Resolution:** Scale connection pool, add PgBouncer sidecar

### 2. Payment Provider Down
**Symptoms:** 500s from Stripe/Braintree, payment failures
**Detection:** Provider health check endpoint failing
**Triage:**
1. Check provider status pages
2. Enable fallback provider (Braintree → Stripe)
3. Increase retry delays
**Resolution:** Wait for provider recovery, drain pending retries

### 3. Fraud Detection False Positives
**Symptoms:** Legitimate payments declined, customer complaints
**Triage:**
1. Query fraud scoring: `fraud/score/{txn_id}`
2. Adjust sensitivity in fraud rules engine
3. Whitelist known-good patterns
**Resolution:** Re-process declined legitimate payments

## Monitoring
- **Error rate:** < 1% (5m avg)
- **P99 latency:** < 500ms
- **Fraud false positive rate:** < 0.1%

## Escalation
1. **L1:** On-call engineer — 15min response
2. **L2:** Payments Squad — 30min
3. **L3:** Engineering Manager (@grace) — 1hr

## Post-Incident Checklist
- [ ] Root cause identified and documented
- [ ] Timeline accurate
- [ ] Monitoring improved if gap found
- [ ] Runbook updated
- [ ] Blameless postmortem posted in #payments-squad
