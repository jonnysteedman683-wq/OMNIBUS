import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { OmniSwarmAdapter } from '../lib/neurocore-swarm';
import { SwarmUpgradeRegistry } from '../lib/upgrade-manifest';

describe('Neurocore Maintenance Agent Verification & Execution', () => {
  it('1. Connects OmniSwarmAdapter and retrieves health metrics', async () => {
    const adapter = new OmniSwarmAdapter();
    await adapter.connect();
    const health = await adapter.getHealthMetrics();
    assert.equal(health.overallStatus, 'healthy');
    assert.ok(health.components.length > 0);
  });

  it('2. Runs system health diagnostics via SwarmUpgradeRegistry', async () => {
    const registry = new SwarmUpgradeRegistry();
    const report = await registry.runHealthDiagnostics();
    assert.equal(report.overallStatus, 'healthy');
    assert.ok(report.components.length >= 5);
  });

  it('3. Executes scheduled upgrade tasks and verifies pending queue clearing', async () => {
    const registry = new SwarmUpgradeRegistry();
    const initialReport = await registry.runHealthDiagnostics();
    const pendingCount = initialReport.pendingUpgradesCount;

    assert.ok(pendingCount > 0, 'Expected pending upgrades to be present');

    const results = await registry.runScheduledUpgrades();
    assert.equal(results.executedCount, pendingCount);
    assert.equal(results.successCount, pendingCount);

    const postReport = await registry.runHealthDiagnostics();
    assert.equal(postReport.pendingUpgradesCount, 0);
    assert.equal(postReport.overallStatus, 'healthy');
  });
});
