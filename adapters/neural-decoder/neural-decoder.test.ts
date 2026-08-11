import { describe, it } from 'node:test';
import assert from 'node:assert';
import { decodeIntent } from './index';

describe('Neural Decoder Adapter', () => {
  it('decodes non-empty intent with positive confidence', () => {
    const out = decodeIntent('diagnostics');
    assert.strictEqual(out.intent, 'diagnostics');
    assert.ok(out.confidence > 0);
    assert.strictEqual(out.features.source, 'neural-decoder');
  });

  it('falls back to noop for empty input', () => {
    const out = decodeIntent('   ');
    assert.strictEqual(out.intent, 'noop');
    assert.strictEqual(out.confidence, 0);
  });
});
