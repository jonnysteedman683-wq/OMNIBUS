import { describe, it } from 'node:test';
import assert from 'node:assert';
import { SwarmUpgradeRegistry } from '../lib/upgrade-manifest.ts';

describe('SwarmUpgradeRegistry Infrastructure Tests', () => {
  it('should run health diagnostics across default registered components', async () => {
    const registry = new SwarmUpgradeRegistry();
    const report = await registry.runHealthDiagnostics();

    assert.strictEqual(report.overallStatus, 'healthy');
    assert.ok(report.components.length >= 5, 'Health report should cover default components including p2p-consensus-engine');
    assert.ok(report.components.some(c => c.component === 'hermes-provider'), 'hermes-provider must be included in health report');
    assert.ok(report.components.some(c => c.component === 'p2p-consensus-engine'), 'p2p-consensus-engine must be included in health report');
    assert.ok(report.pendingUpgradesCount >= 2, 'Default upgrade registry should have pending upgrade tasks');
  });

  it('should execute pending upgrade tasks successfully', async () => {
    const registry = new SwarmUpgradeRegistry();
    const pendingBefore = registry.getPendingTasks();
    assert.ok(pendingBefore.length > 0);

    const taskId = pendingBefore[0].id;
    const result = await registry.executeTask(taskId);

    assert.strictEqual(result.success, true);
    assert.ok(result.logs.length > 0);

    const taskAfter = registry.getTask(taskId);
    assert.strictEqual(taskAfter?.status, 'completed');
  });

  it('should run all scheduled upgrades in batch mode', async () => {
    const registry = new SwarmUpgradeRegistry();
    const batchRes = await registry.runScheduledUpgrades();

    assert.ok(batchRes.executedCount >= 2);
    assert.strictEqual(batchRes.successCount, batchRes.executedCount);
    assert.strictEqual(registry.getPendingTasks().length, 0, 'No pending tasks should remain after scheduled run');
  });
});
