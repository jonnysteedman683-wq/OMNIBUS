import { describe, it } from 'node:test';
import assert from 'node:assert';
import { NeuroStreamAdapter } from '../lib/neurostream-adapter.ts';

describe('NeuroStreamAdapter Integration Tests', () => {
  it('should analyze micro-states and generate structured NeuroIntent', () => {
    const adapter = new NeuroStreamAdapter({ source: 'mock', confidenceThreshold: 0.8 });

    // Push 3 mock telemetry frames
    adapter.pushFrame(NeuroStreamAdapter.createMockFrame(0.95));
    adapter.pushFrame(NeuroStreamAdapter.createMockFrame(0.92));
    adapter.pushFrame(NeuroStreamAdapter.createMockFrame(0.96));

    const microState = adapter.analyzeMicroState();
    assert.ok(microState.alphaPower > 0, 'Alpha power should be positive');
    assert.ok(microState.betaAlphaRatio > 0, 'Beta/Alpha ratio should be calculated');
    assert.ok(microState.confidence >= 0.8, 'Confidence should reflect high quality frames');

    const intent = adapter.generateIntent('route_neural_path', { extraTag: 'test' });
    assert.strictEqual(intent.source, 'mock');
    assert.strictEqual(intent.intent, 'route_neural_path');
    assert.strictEqual(intent.requiresConfirmation, false, 'High confidence should not require confirmation');
    assert.strictEqual(intent.features.extraTag, 'test');
  });

  it('should set requiresConfirmation=true when signal quality is low', () => {
    const adapter = new NeuroStreamAdapter({ source: 'bci', confidenceThreshold: 0.8 });
    
    // Push low quality frame
    adapter.pushFrame(NeuroStreamAdapter.createMockFrame(0.4));

    const intent = adapter.generateIntent('execute');
    assert.strictEqual(intent.requiresConfirmation, true, 'Low signal quality should trigger operator confirmation requirement');
  });
});
