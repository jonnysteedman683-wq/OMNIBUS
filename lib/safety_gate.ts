import { type NeuroIntent } from './neurocore-swarm.ts';

export interface SafetyPolicy {
  allowedIntents: string[];
  blockedIntents: string[];
  confidenceThreshold: number;  // e.g. 0.7
  maxCostUSD: number;           // e.g. 5.0
  allowedFeatures: string[];    // ["alpha_power", "beta_alpha_ratio", "asymmetry", "quality"]
  minQualityThreshold?: number; // e.g. 0.3 (biometric dead-man switch)
  maxRatePerMin?: number;       // default 60
}

export interface PolicyDecision {
  allowed: boolean;
  reason: string;
  requiresOperator: boolean;
  riskLevel: "low" | "moderate" | "high";
}

export const DEFAULT_SAFETY_POLICY: SafetyPolicy = {
  allowedIntents: ["route", "execute", "query", "observe", "route_neural_path", "knowledge_extraction", "task_1", "task_2"],
  blockedIntents: ["override_system", "destructive_action", "unauthorized_command"],
  confidenceThreshold: 0.7,
  maxCostUSD: 5.0,
  allowedFeatures: ["alpha_power", "beta_alpha_ratio", "asymmetry", "quality"],
  minQualityThreshold: 0.3,
  maxRatePerMin: 60
};

export class SafetyGate {
  private policy: SafetyPolicy;
  private intentTimestamps: number[] = [];

  constructor(policy?: Partial<SafetyPolicy>) {
    this.policy = {
      ...DEFAULT_SAFETY_POLICY,
      ...policy
    };
  }

  evaluate(intent: NeuroIntent): PolicyDecision {
    const threshold = this.policy.confidenceThreshold ?? 0.7;

    // Rule 0: Biometric signal quality dead-man switch
    if (intent.features && typeof intent.features.quality === 'number') {
      const minQuality = this.policy.minQualityThreshold ?? 0.3;
      if (intent.features.quality < minQuality) {
        return {
          allowed: false,
          reason: "degraded_signal_quality",
          requiresOperator: true,
          riskLevel: "high"
        };
      }
    }

    // Rule 1: Low confidence check
    if (intent.confidence < threshold) {
      return {
        allowed: false,
        reason: "low_confidence",
        requiresOperator: true,
        riskLevel: "high"
      };
    }

    // Rule 2: Explicitly blocked intent check
    if (this.policy.blockedIntents.includes(intent.intent)) {
      return {
        allowed: false,
        reason: "blocked_intent",
        requiresOperator: true,
        riskLevel: "high"
      };
    }

    // Rule 3: Unknown / disallowed intent check
    if (!this.policy.allowedIntents.includes(intent.intent)) {
      return {
        allowed: false,
        reason: "unknown_intent",
        requiresOperator: true,
        riskLevel: "high"
      };
    }

    // Rule 4: Disallowed feature keys check
    if (intent.features && typeof intent.features === 'object') {
      const allowedFeaturesSet = new Set(this.policy.allowedFeatures);
      const featureKeys = Object.keys(intent.features);
      const hasDisallowedKey = featureKeys.some(key => !allowedFeaturesSet.has(key));

      if (hasDisallowedKey) {
        return {
          allowed: false,
          reason: "feature_blocked",
          requiresOperator: true,
          riskLevel: "high"
        };
      }
    }

    // Rule 5: Rate Limiting Check
    const maxRate = this.policy.maxRatePerMin ?? 60;
    const now = Date.now();
    this.intentTimestamps = this.intentTimestamps.filter(t => now - t < 60000);
    if (this.intentTimestamps.length >= maxRate) {
      return {
        allowed: false,
        reason: "rate_limit_exceeded",
        requiresOperator: true,
        riskLevel: "high"
      };
    }
    this.intentTimestamps.push(now);

    // Rule 6: Moderate risk check (0.7 <= confidence < 0.9 or requiresConfirmation flag)
    const requiresOperator = intent.confidence < 0.9 || intent.requiresConfirmation;
    if (requiresOperator) {
      return {
        allowed: true,
        reason: "allowed_operator_required",
        requiresOperator: true,
        riskLevel: "moderate"
      };
    }

    // Rule 7: Fully allowed
    return {
      allowed: true,
      reason: "allowed",
      requiresOperator: false,
      riskLevel: "low"
    };
  }
}
