import { describe, it } from 'node:test';
import assert from 'node:assert';
import { SafetyGate, type SafetyPolicy } from '../lib/safety_gate.ts';
import { type NeuroIntent } from '../lib/neurocore-swarm.ts';

describe('SafetyGate Policy Enforcement Tests', () => {
  const policy: SafetyPolicy = {
    allowedIntents: ['route', 'execute', 'query', 'observe'],
    blockedIntents: ['override_system', 'destructive_action'],
    confidenceThreshold: 0.7,
    maxCostUSD: 5.0,
    allowedFeatures: ['alpha_power', 'beta_alpha_ratio', 'asymmetry', 'quality']
  };

  const gate = new SafetyGate(policy);

  it('1. should block intents with confidence < 0.7 with reason low_confidence and riskLevel high', () => {
    const intent: NeuroIntent = {
      id: 'intent-low-conf',
      source: 'eeg',
      intent: 'route',
      confidence: 0.65,
      features: { alpha_power: 12.0 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, false);
    assert.strictEqual(decision.reason, 'low_confidence');
    assert.strictEqual(decision.requiresOperator, true);
    assert.strictEqual(decision.riskLevel, 'high');
  });

  it('2. should require operator and set riskLevel moderate for confidence between 0.7 and 0.9', () => {
    const intent: NeuroIntent = {
      id: 'intent-mod-conf',
      source: 'mock',
      intent: 'route',
      confidence: 0.82,
      features: { quality: 0.95 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, true);
    assert.strictEqual(decision.requiresOperator, true);
    assert.strictEqual(decision.riskLevel, 'moderate');
  });

  it('3. should block intent explicitly listed in blockedIntents', () => {
    const intent: NeuroIntent = {
      id: 'intent-blocked',
      source: 'bci',
      intent: 'override_system',
      confidence: 0.95,
      features: { alpha_power: 10.0 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, false);
    assert.strictEqual(decision.reason, 'blocked_intent');
    assert.strictEqual(decision.riskLevel, 'high');
  });

  it('4. should block intent not listed in allowedIntents with reason unknown_intent', () => {
    const intent: NeuroIntent = {
      id: 'intent-unknown',
      source: 'audio',
      intent: 'unregistered_command',
      confidence: 0.95,
      features: { asymmetry: 1.2 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, false);
    assert.strictEqual(decision.reason, 'unknown_intent');
    assert.strictEqual(decision.riskLevel, 'high');
  });

  it('5. should block intents containing feature keys not in allowedFeatures', () => {
    const intent: NeuroIntent = {
      id: 'intent-disallowed-feature',
      source: 'eeg',
      intent: 'execute',
      confidence: 0.92,
      features: { alpha_power: 10.0, raw_audio_payload: 'unauthorized_data' },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, false);
    assert.strictEqual(decision.reason, 'feature_blocked');
    assert.strictEqual(decision.riskLevel, 'high');
  });

  it('6. should allow valid intent with confidence >= 0.9 with riskLevel low', () => {
    const intent: NeuroIntent = {
      id: 'intent-valid',
      source: 'eeg',
      intent: 'observe',
      confidence: 0.98,
      features: { alpha_power: 15.2, beta_alpha_ratio: 0.45 },
      timestamp: Date.now(),
      requiresConfirmation: false
    };

    const decision = gate.evaluate(intent);
    assert.strictEqual(decision.allowed, true);
    assert.strictEqual(decision.reason, 'allowed');
    assert.strictEqual(decision.requiresOperator, false);
    assert.strictEqual(decision.riskLevel, 'low');
  });
});
