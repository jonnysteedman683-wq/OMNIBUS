import { describe, it } from 'node:test';
import assert from 'node:assert';
import { FederatedDebateEngine } from '../lib/swarm-debate.ts';
import { NeuroStreamAdapter } from '../lib/neurostream-adapter.ts';
import { OmniSwarmAdapter, type NeuroIntent } from '../lib/neurocore-swarm.ts';

describe('FederatedDebateEngine Unit & Integration Tests', () => {
  it('should adapt consensus threshold dynamically based on neural beta/alpha ratio', () => {
    const engine = new FederatedDebateEngine({ enableNeuralAdaptation: true, baseThreshold: 0.85 });

    // High focus state (beta/alpha ratio >= 1.8) -> lower threshold 0.75
    const highFocusState = { alphaPower: 5.0, betaPower: 10.0, betaAlphaRatio: 2.0, asymmetry: 0, quality: 1.0, confidence: 0.95 };
    assert.strictEqual(engine.calculateAdaptiveThreshold(highFocusState), 0.75);

    // Baseline focus state (1.0 <= beta/alpha ratio < 1.8) -> standard threshold 0.85
    const normalState = { alphaPower: 10.0, betaPower: 12.0, betaAlphaRatio: 1.2, asymmetry: 0, quality: 1.0, confidence: 0.90 };
    assert.strictEqual(engine.calculateAdaptiveThreshold(normalState), 0.85);

    // Low focus / drowsiness state (beta/alpha ratio < 1.0) -> tightened threshold 0.92
    const lowFocusState = { alphaPower: 15.0, betaPower: 8.0, betaAlphaRatio: 0.53, asymmetry: 0, quality: 0.7, confidence: 0.70 };
    assert.strictEqual(engine.calculateAdaptiveThreshold(lowFocusState), 0.92);
  });

  it('should run a 3-round Triad debate and produce an auto_execute verdict for high-confidence intent', async () => {
    const engine = new FederatedDebateEngine();

    const intent: NeuroIntent = {
      id: 'intent-debate-01',
      source: 'eeg',
      intent: 'route_neural_path',
      confidence: 0.95,
      features: { skillId: 'skill-1' },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const result = await engine.runDebate(intent);

    assert.strictEqual(result.topic, 'route_neural_path');
    assert.strictEqual(result.rounds.length, 3, 'Debate should execute 3 triad rounds');
    assert.strictEqual(result.verdict, 'auto_execute');
    assert.strictEqual(result.winningSkillId, 'skill-1');
    assert.ok(result.consensusScore >= result.requiredThreshold);
  });

  it('should require operator confirmation when low confidence flags risk in triad debate', async () => {
    const engine = new FederatedDebateEngine();

    const intent: NeuroIntent = {
      id: 'intent-debate-risk',
      source: 'bci',
      intent: 'query',
      confidence: 0.60,
      features: {},
      timestamp: Date.now(),
      requiresConfirmation: true
    };

    const result = await engine.runDebate(intent);

    assert.strictEqual(result.verdict, 'operator_confirmation_required');
    assert.ok(result.consensusScore < result.requiredThreshold);
  });

  it('should run debate via OmniSwarmAdapter.runDebate()', async () => {
    const adapter = new OmniSwarmAdapter();
    await adapter.connect();

    const intent: NeuroIntent = {
      id: 'intent-adapter-debate',
      source: 'mock',
      intent: 'knowledge_extraction',
      confidence: 0.92,
      features: { skillId: 'skill-2' },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const debateResult = await adapter.runDebate(intent);

    assert.ok(debateResult.debateId);
    assert.strictEqual(debateResult.verdict, 'auto_execute');
    assert.strictEqual(debateResult.winningSkillId, 'skill-2');
  });
});
