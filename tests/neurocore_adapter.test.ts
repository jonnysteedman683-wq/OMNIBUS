import { describe, it } from 'node:test';
import assert from 'node:assert';
import { OmniSwarmAdapter, type NeuroIntent } from '../lib/neurocore-swarm.ts';

describe('OmniSwarmAdapter Integration Tests', () => {
  it('should complete the connect -> capabilities -> start -> status -> stop lifecycle', async () => {
    const adapter = new OmniSwarmAdapter();

    // 1. Connect
    await adapter.connect({ maxConcurrency: 5 });

    // 2. Capabilities
    const caps = await adapter.capabilities();
    assert.ok(Array.isArray(caps.roles), 'Capabilities should return roles array');
    assert.ok(caps.roles.includes('Neural Path Routing'), 'Roles should contain default skills');
    assert.deepStrictEqual(caps.phases, ['planning', 'execution', 'verification', 'debate']);

    // 3. Start (with requiresConfirmation=false to test auto processing in lifecycle)
    const intent: NeuroIntent = {
      id: 'intent-lifecycle-01',
      source: 'eeg',
      intent: 'route_neural_path',
      confidence: 0.95,
      features: { band: 'alpha', power: 12.5 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const startRes = await adapter.start(intent);
    assert.ok(startRes.actionId, 'Action ID should be returned');
    assert.strictEqual(startRes.status, 'completed', 'Status should be completed');

    // 4. Status (query completed action)
    const statusRes = await adapter.status(startRes.actionId);
    assert.strictEqual(statusRes.actionId, startRes.actionId);
    assert.strictEqual(statusRes.status, 'completed');
    assert.ok(Array.isArray(statusRes.output));

    // 5. Stop (querying non-existent/processed pending action)
    const stopRes = await adapter.stop(startRes.actionId);
    assert.strictEqual(stopRes.actionId, startRes.actionId);
    assert.strictEqual(stopRes.status, 'not_found');
  });

  it('should return pending_confirmation when requiresConfirmation=true', async () => {
    const adapter = new OmniSwarmAdapter();
    await adapter.connect();

    const intent: NeuroIntent = {
      id: 'intent-confirm-true',
      source: 'mock',
      intent: 'high_risk_override',
      confidence: 0.65,
      features: { risk: 'high' },
      timestamp: Date.now(),
      requiresConfirmation: true
    };

    const startRes = await adapter.start(intent);
    assert.ok(startRes.actionId);
    assert.strictEqual(startRes.status, 'pending_confirmation');

    // Action should remain in pending state and not processed
    const statusRes = await adapter.status(startRes.actionId);
    assert.strictEqual(statusRes.status, 'pending');
  });

  it('should process queue immediately when requiresConfirmation=false', async () => {
    const adapter = new OmniSwarmAdapter();
    await adapter.connect();

    const intent: NeuroIntent = {
      id: 'intent-auto-process',
      source: 'audio',
      intent: 'knowledge_extraction',
      confidence: 0.98,
      features: { query: 'extract entities' },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const startRes = await adapter.start(intent);
    assert.ok(startRes.actionId);
    assert.strictEqual(startRes.status, 'completed');

    // Status query should return completed
    const statusRes = await adapter.status(startRes.actionId);
    assert.strictEqual(statusRes.status, 'completed');
    assert.ok(statusRes.output.some(line => line.includes('Status: completed')));
  });

  it('should clear all pending actions on emergencyStop()', async () => {
    const adapter = new OmniSwarmAdapter();
    await adapter.connect();

    const intent1: NeuroIntent = {
      id: 'intent-em-1',
      source: 'bci',
      intent: 'task_1',
      confidence: 0.7,
      features: {},
      timestamp: Date.now(),
      requiresConfirmation: true
    };
    const intent2: NeuroIntent = {
      id: 'intent-em-2',
      source: 'bci',
      intent: 'task_2',
      confidence: 0.7,
      features: {},
      timestamp: Date.now(),
      requiresConfirmation: true
    };

    const res1 = await adapter.start(intent1);
    const res2 = await adapter.start(intent2);

    assert.strictEqual((await adapter.status(res1.actionId)).status, 'pending');
    assert.strictEqual((await adapter.status(res2.actionId)).status, 'pending');

    const emRes = await adapter.emergencyStop();
    assert.strictEqual(emRes.stopped, true);

    // Verify all pending actions are cleared
    assert.strictEqual((await adapter.status(res1.actionId)).status, 'unknown');
    assert.strictEqual((await adapter.status(res2.actionId)).status, 'unknown');
  });
});
