import { test } from 'node:test';
import assert from 'node:assert/strict';
import { HermesAgentAdapter, type HermesConfig } from '../lib/hermes-adapter.ts';
import { OmniSwarmAdapter, type NeuroIntent } from '../lib/neurocore-swarm.ts';

test('HermesAgentAdapter - connect & capabilities in offline fallback mode', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });
  
  assert.equal(adapter.isConnected(), true);
  assert.equal(adapter.isOffline(), true);

  const caps = await adapter.capabilities();
  assert.equal(caps.provider, 'hermes-local');
  assert.equal(caps.freeTier, true);
  assert.ok(caps.roles.includes('architect'));
  assert.ok(caps.roles.includes('guard'));
  assert.ok(caps.phases.includes('debate'));
});

test('HermesAgentAdapter - start & status intent execution lifecycle', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const intent: NeuroIntent = {
    id: 'intent-hermes-1',
    source: 'mock',
    intent: 'route',
    confidence: 0.95,
    features: { alpha_power: 0.85, quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const startRes = await adapter.start(intent);
  assert.equal(startRes.status, 'completed');
  assert.ok(startRes.actionId.startsWith('hermes-action-'));

  const statusRes = await adapter.status(startRes.actionId);
  assert.equal(statusRes.status, 'completed');
  assert.ok(statusRes.output.length > 0);
  assert.ok(statusRes.output.some(line => line.includes('Zero Cost')));
});

test('HermesAgentAdapter - handles pending_confirmation intents', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const pendingIntent: NeuroIntent = {
    id: 'intent-pending-1',
    source: 'mock',
    intent: 'execute',
    confidence: 0.95,
    features: { quality: 0.8 },
    timestamp: Date.now(),
    requiresConfirmation: true
  };

  const startRes = await adapter.start(pendingIntent);
  assert.equal(startRes.status, 'pending_confirmation');

  const statusRes = await adapter.status(startRes.actionId);
  assert.equal(statusRes.status, 'pending_confirmation');

  const stopRes = await adapter.stop(startRes.actionId);
  assert.equal(stopRes.status, 'cancelled');
});

test('HermesAgentAdapter - SafetyGate enforcement blocks degraded signal quality', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const degradedIntent: NeuroIntent = {
    id: 'intent-degraded-signal',
    source: 'eeg',
    intent: 'route',
    confidence: 0.95,
    features: { quality: 0.1 }, // quality < 0.3 threshold
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  await assert.rejects(
    async () => { await adapter.start(degradedIntent); },
    (err: Error) => {
      return err.message.includes('rejected by SafetyGate') && err.message.includes('degraded_signal_quality');
    }
  );
});

test('HermesAgentAdapter - SafetyGate enforcement blocks low confidence intent', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const lowConfIntent: NeuroIntent = {
    id: 'intent-low-conf',
    source: 'mock',
    intent: 'route',
    confidence: 0.4, // < 0.7 threshold
    features: { quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  await assert.rejects(
    async () => { await adapter.start(lowConfIntent); },
    (err: Error) => {
      return err.message.includes('rejected by SafetyGate') && err.message.includes('low_confidence');
    }
  );
});

test('HermesAgentAdapter - SafetyGate enforcement blocks explicitly blocked intents', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const blockedIntent: NeuroIntent = {
    id: 'intent-blocked',
    source: 'mock',
    intent: 'override_system',
    confidence: 0.99,
    features: { quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  await assert.rejects(
    async () => { await adapter.start(blockedIntent); },
    (err: Error) => {
      return err.message.includes('rejected by SafetyGate') && err.message.includes('blocked_intent');
    }
  );
});

test('HermesAgentAdapter - emergencyStop halts and purges pending actions', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const pendingIntent: NeuroIntent = {
    id: 'intent-emergency-1',
    source: 'mock',
    intent: 'query',
    confidence: 0.95,
    features: { quality: 0.8 },
    timestamp: Date.now(),
    requiresConfirmation: true
  };

  const startRes = await adapter.start(pendingIntent);
  assert.equal(startRes.status, 'pending_confirmation');

  const result = await adapter.emergencyStop();
  assert.equal(result.stopped, true);

  const statusRes = await adapter.status(startRes.actionId);
  assert.equal(statusRes.status, 'cancelled');
});

test('OmniSwarmAdapter - provider registration and emergencyStop cascade', async () => {
  const omni = new OmniSwarmAdapter();
  await omni.connect();

  const hermes = new HermesAgentAdapter();
  await hermes.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  omni.registerProvider('hermes', hermes);
  assert.equal(omni.getProvider('hermes'), hermes);
  assert.equal(omni.getProviders().size, 2);

  const pendingIntent: NeuroIntent = {
    id: 'intent-cascade-1',
    source: 'mock',
    intent: 'observe',
    confidence: 0.95,
    features: { quality: 0.8 },
    timestamp: Date.now(),
    requiresConfirmation: true
  };

  const hermesAction = await hermes.start(pendingIntent);
  assert.equal(hermesAction.status, 'pending_confirmation');

  const omniEmergencyRes = await omni.emergencyStop();
  assert.equal(omniEmergencyRes.stopped, true);

  const statusRes = await hermes.status(hermesAction.actionId);
  assert.equal(statusRes.status, 'cancelled');
});
