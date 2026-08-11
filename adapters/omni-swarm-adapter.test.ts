import assert from 'assert';
import { OmniSwarmAdapter, type NeuroIntent } from '../lib/neurocore-swarm.ts';

/**
 * Mock implementation of SwarmRuntimeEngine as required for adapter testing
 */
class MockSwarmRuntimeEngine {
  actionQueue: any[] = [];
  skillCards: Map<string, any> = new Map([
    ['skill-1', { id: 'skill-1', name: 'Neural Path Routing' }],
    ['skill-2', { id: 'skill-2', name: 'Knowledge Extraction' }]
  ]);
  enqueueActionCalled = false;
  processQueueCalled = false;

  enqueueAction(type: string, target: string, parameters: Record<string, unknown> = {}, reasoning: string = '') {
    this.enqueueActionCalled = true;
    const action = {
      id: `action-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      timestamp: Date.now(),
      type,
      target,
      parameters,
      reasoning,
      status: 'pending'
    };
    this.actionQueue.push(action);
    return action;
  }

  async processQueue(callback?: (action: any) => void) {
    this.processQueueCalled = true;
    const processed: any[] = [];
    while (this.actionQueue.length > 0) {
      const action = this.actionQueue.shift();
      action.status = 'completed';
      processed.push(action);
      if (typeof callback === 'function') {
        callback(action);
      }
    }
    return processed;
  }
}

async function testSuite() {
  console.log('--- Running OmniSwarmAdapter Test Suite ---');

  // -------------------------------------------------------------
  // Test 1: connect() successfully initializes the engine
  // -------------------------------------------------------------
  const adapter = new OmniSwarmAdapter();
  await adapter.connect({ baseUrl: 'http://localhost:3000' });
  const internalEngine = (adapter as any).engine;
  assert.ok(internalEngine, 'Engine should be initialized after connect()');
  console.log('✔ Test 1 passed: connect() successfully initializes engine');

  // Inject MockSwarmRuntimeEngine to verify behavior deterministically
  const mockEngine = new MockSwarmRuntimeEngine();
  (adapter as any).engine = mockEngine;

  // -------------------------------------------------------------
  // Test 2: capabilities() returns roles + phases
  // -------------------------------------------------------------
  const caps = await adapter.capabilities();
  assert.ok(Array.isArray(caps.roles), 'Capabilities should return roles array');
  assert.strictEqual(caps.roles.length, 2, 'Roles length should match mock skill cards');
  assert.deepStrictEqual(caps.roles, ['Neural Path Routing', 'Knowledge Extraction']);
  assert.deepStrictEqual(caps.phases, ['planning', 'execution', 'verification']);
  console.log('✔ Test 2 passed: capabilities() returns roles + phases');

  // -------------------------------------------------------------
  // Test 3: start() with requiresConfirmation=false
  // enqueueAction is called, processQueue runs, action status is "completed"
  // -------------------------------------------------------------
  const autoIntent: NeuroIntent = {
    id: 'intent-auto-01',
    source: 'eeg',
    intent: 'route_neural_path',
    confidence: 0.95,
    features: { speed: 1.0 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  mockEngine.enqueueActionCalled = false;
  mockEngine.processQueueCalled = false;

  const startResAuto = await adapter.start(autoIntent);
  assert.ok(startResAuto.actionId, 'Action ID should be returned');
  assert.strictEqual(startResAuto.status, 'completed', 'Status should be completed when requiresConfirmation=false');
  assert.strictEqual(mockEngine.enqueueActionCalled, true, 'enqueueAction should be called');
  assert.strictEqual(mockEngine.processQueueCalled, true, 'processQueue should be called');
  console.log('✔ Test 3 passed: start() with requiresConfirmation=false completes action');

  // -------------------------------------------------------------
  // Test 4: start() with requiresConfirmation=true
  // action status is "pending_confirmation", no processing
  // -------------------------------------------------------------
  const confirmIntent: NeuroIntent = {
    id: 'intent-confirm-02',
    source: 'mock',
    intent: 'destructive_action',
    confidence: 0.70,
    features: { risk: 'high' },
    timestamp: Date.now(),
    requiresConfirmation: true
  };

  mockEngine.processQueueCalled = false;

  const startResConfirm = await adapter.start(confirmIntent);
  assert.ok(startResConfirm.actionId, 'Action ID should be returned');
  assert.strictEqual(startResConfirm.status, 'pending_confirmation', 'Status should be pending_confirmation');
  assert.strictEqual(mockEngine.processQueueCalled, false, 'processQueue should NOT be called for confirmation intents');
  console.log('✔ Test 4 passed: start() with requiresConfirmation=true returns pending_confirmation');

  // -------------------------------------------------------------
  // Test 5: status() returns correct status and output for a processed action
  // -------------------------------------------------------------
  const statusRes = await adapter.status(startResAuto.actionId);
  assert.strictEqual(statusRes.actionId, startResAuto.actionId);
  assert.strictEqual(statusRes.status, 'completed');
  assert.ok(Array.isArray(statusRes.output), 'Output should be an array');
  assert.ok(statusRes.output.some(line => line.includes('Status: completed')), 'Output should contain completed status');
  console.log('✔ Test 5 passed: status() returns correct status and output for processed action');

  // -------------------------------------------------------------
  // Test 6: stop() removes a pending action
  // -------------------------------------------------------------
  const stopRes = await adapter.stop(startResConfirm.actionId);
  assert.strictEqual(stopRes.actionId, startResConfirm.actionId);
  assert.strictEqual(stopRes.status, 'cancelled', 'Pending action should be cancelled upon stop()');

  const statusAfterStop = await adapter.status(startResConfirm.actionId);
  assert.strictEqual(statusAfterStop.status, 'unknown', 'Action status should be unknown after being stopped');
  console.log('✔ Test 6 passed: stop() removes pending action');

  // -------------------------------------------------------------
  // Test 7: emergencyStop() clears all pending actions and returns stopped=true
  // -------------------------------------------------------------
  const pending1: NeuroIntent = {
    id: 'p1', source: 'audio', intent: 'cmd1', confidence: 0.8,
    features: {}, timestamp: Date.now(), requiresConfirmation: true
  };
  const pending2: NeuroIntent = {
    id: 'p2', source: 'audio', intent: 'cmd2', confidence: 0.8,
    features: {}, timestamp: Date.now(), requiresConfirmation: true
  };

  const act1 = await adapter.start(pending1);
  const act2 = await adapter.start(pending2);

  const emStopRes = await adapter.emergencyStop();
  assert.strictEqual(emStopRes.stopped, true, 'emergencyStop should return stopped=true');

  const check1 = await adapter.status(act1.actionId);
  const check2 = await adapter.status(act2.actionId);
  assert.strictEqual(check1.status, 'unknown', 'Action 1 should be cleared after emergencyStop');
  assert.strictEqual(check2.status, 'unknown', 'Action 2 should be cleared after emergencyStop');
  console.log('✔ Test 7 passed: emergencyStop() clears all pending actions');

  console.log('--- All OmniSwarmAdapter Tests Passed Successfully! ---');
}

testSuite().catch((err) => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
