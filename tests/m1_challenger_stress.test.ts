import { test } from 'node:test';
import assert from 'node:assert/strict';
import { HermesAgentAdapter } from '../lib/hermes-adapter.ts';
import { OmniSwarmAdapter, type NeuroIntent } from '../lib/neurocore-swarm.ts';

test('CHALLENGER STRESS 1: Rapid Intent Dispatches & Rate Limiting Threshold Breach', async () => {
  const adapter = new HermesAgentAdapter({
    safetyPolicy: { maxRatePerMin: 10 }
  });
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  let successCount = 0;
  let rateLimitedCount = 0;

  for (let i = 0; i < 15; i++) {
    const intent: NeuroIntent = {
      id: `rapid-intent-${i}`,
      source: 'mock',
      intent: 'route',
      confidence: 0.95,
      features: { quality: 0.9 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    try {
      const res = await adapter.start(intent);
      if (res.status === 'completed') {
        successCount++;
      }
    } catch (err: any) {
      if (err.message.includes('rate_limit_exceeded')) {
        rateLimitedCount++;
      } else {
        throw err;
      }
    }
  }

  assert.equal(successCount, 10, 'Exactly 10 dispatches should succeed before hitting maxRatePerMin limit of 10');
  assert.equal(rateLimitedCount, 5, 'Exactly 5 dispatches should be rejected with rate_limit_exceeded');
});

test('CHALLENGER STRESS 2: Degraded Biometric Signal Quality Boundary Checks', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const qualityValues = [0.1, 0.29, 0.0, -1.0];
  for (const q of qualityValues) {
    const intent: NeuroIntent = {
      id: `intent-quality-${q}`,
      source: 'eeg',
      intent: 'route',
      confidence: 0.95,
      features: { quality: q },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    await assert.rejects(
      async () => { await adapter.start(intent); },
      (err: Error) => {
        return err.message.includes('rejected by SafetyGate') && err.message.includes('degraded_signal_quality');
      },
      `Quality ${q} should be rejected due to degraded signal quality`
    );
  }

  // Boundary condition: quality = 0.3 (should pass minQualityThreshold of 0.3)
  const validQualityIntent: NeuroIntent = {
    id: 'intent-quality-0.3',
    source: 'eeg',
    intent: 'route',
    confidence: 0.95,
    features: { quality: 0.3 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };
  const res = await adapter.start(validQualityIntent);
  assert.equal(res.status, 'completed');
});

test('CHALLENGER STRESS 3: Blocked & Malicious Intent Payloads', async () => {
  const adapter = new HermesAgentAdapter();
  await adapter.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const blockedIntents = ['override_system', 'destructive_action', 'unauthorized_command'];
  for (const blocked of blockedIntents) {
    const intent: NeuroIntent = {
      id: `blocked-${blocked}`,
      source: 'mock',
      intent: blocked,
      confidence: 0.99,
      features: { quality: 0.9 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    await assert.rejects(
      async () => { await adapter.start(intent); },
      (err: Error) => {
        return err.message.includes('rejected by SafetyGate') && err.message.includes('blocked_intent');
      },
      `Intent '${blocked}' must be blocked by SafetyGate`
    );
  }

  // Unknown/Disallowed Intent
  const unknownIntent: NeuroIntent = {
    id: 'unknown-intent-1',
    source: 'mock',
    intent: 'arbitrary_code_execution',
    confidence: 0.99,
    features: { quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  await assert.rejects(
    async () => { await adapter.start(unknownIntent); },
    (err: Error) => {
      return err.message.includes('rejected by SafetyGate') && err.message.includes('unknown_intent');
    }
  );
});

test('CHALLENGER STRESS 4: EmergencyStop Map Clearing & Multi-Provider Cascade', async () => {
  const omni = new OmniSwarmAdapter();
  await omni.connect();

  const hermes1 = new HermesAgentAdapter();
  await hermes1.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  const hermes2 = new HermesAgentAdapter();
  await hermes2.connect({ baseUrl: 'http://127.0.0.1:59999', timeoutMs: 100 });

  omni.registerProvider('hermes1', hermes1);
  omni.registerProvider('hermes2', hermes2);

  // Enqueue pending confirmation actions on both adapters
  const pendingActionsHermes1: string[] = [];
  const pendingActionsHermes2: string[] = [];

  for (let i = 0; i < 5; i++) {
    const intent1: NeuroIntent = {
      id: `h1-pending-${i}`,
      source: 'mock',
      intent: 'execute',
      confidence: 0.95,
      features: { quality: 0.9 },
      timestamp: Date.now(),
      requiresConfirmation: true
    };
    const res1 = await hermes1.start(intent1);
    pendingActionsHermes1.push(res1.actionId);

    const intent2: NeuroIntent = {
      id: `h2-pending-${i}`,
      source: 'mock',
      intent: 'query',
      confidence: 0.95,
      features: { quality: 0.9 },
      timestamp: Date.now(),
      requiresConfirmation: true
    };
    const res2 = await hermes2.start(intent2);
    pendingActionsHermes2.push(res2.actionId);
  }

  // Confirm pending status
  for (const id of pendingActionsHermes1) {
    const st = await hermes1.status(id);
    assert.equal(st.status, 'pending_confirmation');
  }
  for (const id of pendingActionsHermes2) {
    const st = await hermes2.status(id);
    assert.equal(st.status, 'pending_confirmation');
  }

  // Trigger OmniSwarmAdapter Emergency Stop (Cascaded)
  const stopResult = await omni.emergencyStop();
  assert.equal(stopResult.stopped, true, 'Emergency stop should return stopped: true');

  // Verify all pending actions across both providers were cancelled and cleared from pending map
  for (const id of pendingActionsHermes1) {
    const st = await hermes1.status(id);
    assert.equal(st.status, 'cancelled', `Action ${id} on hermes1 must be cancelled`);
    assert.ok(st.output.some(line => line.includes('Emergency stop invoked')), 'Output should document emergency stop');
  }
  for (const id of pendingActionsHermes2) {
    const st = await hermes2.status(id);
    assert.equal(st.status, 'cancelled', `Action ${id} on hermes2 must be cancelled`);
    assert.ok(st.output.some(line => line.includes('Emergency stop invoked')), 'Output should document emergency stop');
  }

  // Verify new intents can be dispatched cleanly after emergency stop
  const postEmergencyIntent: NeuroIntent = {
    id: 'post-emergency-intent',
    source: 'mock',
    intent: 'route',
    confidence: 0.95,
    features: { quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };
  const postRes = await hermes1.start(postEmergencyIntent);
  assert.equal(postRes.status, 'completed', 'New intents should execute successfully post emergency stop');
});

test('CHALLENGER STRESS 5: OmniSwarmAdapter Integration with EnableHermes Config', async () => {
  const omni = new OmniSwarmAdapter();
  await omni.connect({ enableHermes: true, hermesConfig: { fallbackToMock: true } });

  const hermes = omni.getProvider('hermes');
  assert.ok(hermes, 'Hermes provider should be registered when enableHermes is true');

  const caps = await hermes.capabilities();
  assert.equal(caps.roles.includes('architect'), true);

  const intent: NeuroIntent = {
    id: 'omni-hermes-integration-1',
    source: 'mock',
    intent: 'observe',
    confidence: 0.95,
    features: { quality: 0.9 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const startRes = await hermes.start(intent);
  assert.equal(startRes.status, 'completed');
});
