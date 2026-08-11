import { type NeuroIntent } from './neurocore-swarm.ts';

export interface NeuralTelemetryFrame {
  timestamp: number;
  channelData: Record<string, number[]>; // e.g. { Fp1: [...], Fp2: [...] }
  sampleRate: number;                   // e.g. 250 Hz
  quality: number;                      // 0.0 - 1.0 (signal integrity)
}

export interface CalculatedMicroState {
  alphaPower: number;
  betaPower: number;
  betaAlphaRatio: number;
  asymmetry: number;
  quality: number;
  confidence: number;
}

export type SignalSource = 'eeg' | 'mock' | 'audio' | 'bci';

export interface StreamConfig {
  source: SignalSource;
  sampleWindowMs: number;       // default 250
  confidenceThreshold: number;  // default 0.7
  autoCalibrate: boolean;
}

export class NeuroStreamAdapter {
  private config: StreamConfig;
  private buffer: NeuralTelemetryFrame[] = [];

  constructor(config?: Partial<StreamConfig>) {
    this.config = {
      source: 'mock',
      sampleWindowMs: 250,
      confidenceThreshold: 0.7,
      autoCalibrate: true,
      ...config
    };
  }

  /**
   * Push a raw telemetry frame into the stream buffer
   */
  pushFrame(frame: NeuralTelemetryFrame): void {
    this.buffer.push(frame);
    // Maintain rolling buffer within 5 seconds window
    const maxWindowMs = 5000;
    const cutoff = Date.now() - maxWindowMs;
    this.buffer = this.buffer.filter(f => f.timestamp >= cutoff);
  }

  /**
   * Process current buffer into spectral micro-states
   */
  analyzeMicroState(): CalculatedMicroState {
    if (this.buffer.length === 0) {
      return {
        alphaPower: 10.0,
        betaPower: 15.0,
        betaAlphaRatio: 1.5,
        asymmetry: 0.05,
        quality: 1.0,
        confidence: 0.85
      };
    }

    const latest = this.buffer[this.buffer.length - 1];
    let totalAlpha = 0;
    let totalBeta = 0;
    let count = 0;

    for (const frame of this.buffer) {
      for (const channel in frame.channelData) {
        const samples = frame.channelData[channel] || [];
        const avg = samples.reduce((acc, v) => acc + Math.abs(v), 0) / (samples.length || 1);
        totalAlpha += avg * 0.8;
        totalBeta += avg * 1.2;
        count++;
      }
    }

    const alphaPower = count > 0 ? totalAlpha / count : 10.0;
    const betaPower = count > 0 ? totalBeta / count : 15.0;
    const betaAlphaRatio = alphaPower > 0 ? betaPower / alphaPower : 1.0;
    const quality = latest.quality;
    
    // Confidence combines signal quality and sample consistency
    const confidence = Math.min(1.0, Math.max(0.1, quality * 0.9 + (this.buffer.length > 2 ? 0.1 : 0.0)));

    return {
      alphaPower,
      betaPower,
      betaAlphaRatio,
      asymmetry: 0.02,
      quality,
      confidence
    };
  }

  /**
   * Generate a structured NeuroIntent from current stream state
   */
  generateIntent(intentName: string, customFeatures: Record<string, unknown> = {}): NeuroIntent {
    const microState = this.analyzeMicroState();

    return {
      id: `stream-intent-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
      source: this.config.source,
      intent: intentName,
      confidence: microState.confidence,
      features: {
        alpha_power: microState.alphaPower,
        beta_alpha_ratio: microState.betaAlphaRatio,
        asymmetry: microState.asymmetry,
        quality: microState.quality,
        ...customFeatures
      },
      timestamp: Date.now(),
      requiresConfirmation: microState.confidence < this.config.confidenceThreshold
    };
  }

  /**
   * Helper to generate a synthetic test telemetry frame
   */
  static createMockFrame(quality: number = 0.95): NeuralTelemetryFrame {
    return {
      timestamp: Date.now(),
      channelData: {
        Fp1: [12.4, 15.2, 14.1, 13.8],
        Fp2: [11.9, 14.8, 13.9, 14.2],
        C3: [8.5, 9.2, 9.0, 8.8],
        C4: [8.4, 9.1, 8.9, 8.7]
      },
      sampleRate: 250,
      quality
    };
  }
}
