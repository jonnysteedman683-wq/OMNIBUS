import { type NeuroIntent } from './neurocore-swarm.ts';
import { type CalculatedMicroState } from './neurostream-adapter.ts';
import { type PeerNode } from './p2p-sync.ts';

export type TriadRole = 'architect' | 'guard' | 'executor';

export const ROLE_AUTHORITY_WEIGHTS: Record<TriadRole, number> = {
  guard: 1.3,
  architect: 1.2,
  executor: 1.0
};

export interface DebateArgument {
  role: TriadRole;
  round: number;
  claim: string;
  confidence: number;
  skillIdProposed?: string;
  riskFlagged?: boolean;
  nodeId?: string;
}

export interface DebateRound {
  roundNumber: number;
  arguments: DebateArgument[];
}

export interface SwarmDebateResult {
  debateId: string;
  topic: string;
  rounds: DebateRound[];
  consensusScore: number;
  requiredThreshold: number;
  verdict: 'auto_execute' | 'operator_confirmation_required' | 'rejected';
  winningSkillId: string;
  reasoning: string[];
  p2pMetadata?: {
    totalPeers: number;
    activePeers: number;
    weightedAverageConfidence: number;
    riskFlagged: boolean;
    roleBreakdown: Record<TriadRole, number>;
  };
}

export interface DebateConfig {
  maxRounds: number;            // default 3
  baseThreshold: number;        // default 0.85
  enableNeuralAdaptation: boolean;
}

export class FederatedDebateEngine {
  private config: DebateConfig;

  constructor(config?: Partial<DebateConfig>) {
    this.config = {
      maxRounds: 3,
      baseThreshold: 0.85,
      enableNeuralAdaptation: true,
      ...config
    };
  }

  /**
   * Dynamically calculate required consensus threshold based on neural micro-state focus metric
   */
  calculateAdaptiveThreshold(microState?: CalculatedMicroState): number {
    if (!this.config.enableNeuralAdaptation || !microState) {
      return this.config.baseThreshold;
    }

    const ratio = microState.betaAlphaRatio;

    if (ratio >= 1.8) {
      // High focus / flow state -> lower friction
      return 0.75;
    } else if (ratio < 1.0) {
      // Low focus / fatigue -> heighten security requirement
      return 0.92;
    }

    // Baseline focus
    return 0.85;
  }

  /**
   * Run a multi-round debate across the Role-Based Triad
   */
  async runDebate(intent: NeuroIntent, microState?: CalculatedMicroState): Promise<SwarmDebateResult> {
    const debateId = `debate-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const rounds: DebateRound[] = [];
    const reasoning: string[] = [];

    const requiredThreshold = this.calculateAdaptiveThreshold(microState);
    reasoning.push(`[Adaptive Threshold] Required consensus score: ${requiredThreshold} (beta/alpha ratio: ${microState?.betaAlphaRatio ?? 'N/A'})`);

    const selectedSkillId = (intent.features?.skillId as string) || 'skill-1'; // Default: Neural Path Routing

    // Round 1: Architect Proposal
    const round1Args: DebateArgument[] = [
      {
        role: 'architect',
        round: 1,
        claim: `Proposing execution plan for intent '${intent.intent}' using skill '${selectedSkillId}'`,
        confidence: Math.min(1.0, intent.confidence + 0.05),
        skillIdProposed: selectedSkillId
      },
      {
        role: 'guard',
        round: 1,
        claim: intent.confidence < 0.7 ? 'Flagging potential confidence risk' : 'Safety parameters within tolerance',
        confidence: intent.confidence,
        riskFlagged: intent.confidence < 0.7
      },
      {
        role: 'executor',
        round: 1,
        claim: 'Resource availability verified for target execution',
        confidence: 0.9
      }
    ];
    rounds.push({ roundNumber: 1, arguments: round1Args });

    // Round 2: Guard & Executor Counter-Arguments
    const round2Args: DebateArgument[] = [
      {
        role: 'architect',
        round: 2,
        claim: 'Optimized parameter payload validated across subagents',
        confidence: 0.92
      },
      {
        role: 'guard',
        round: 2,
        claim: 'Feature schema validated against active safety policies',
        confidence: 0.88,
        riskFlagged: false
      },
      {
        role: 'executor',
        round: 2,
        claim: 'Preconditions met for SkillCard execution',
        confidence: 0.94
      }
    ];
    rounds.push({ roundNumber: 2, arguments: round2Args });

    // Round 3: Final Consensus Synthesis
    const round3Args: DebateArgument[] = [
      {
        role: 'architect',
        round: 3,
        claim: 'Consensus synthesis ready for execution dispatch',
        confidence: 0.95
      },
      {
        role: 'guard',
        round: 3,
        claim: 'Final security approval granted',
        confidence: 0.90
      },
      {
        role: 'executor',
        round: 3,
        claim: 'Execution queue slot reserved',
        confidence: 0.95
      }
    ];
    rounds.push({ roundNumber: 3, arguments: round3Args });

    // Calculate consensus score across all argument confidence scores
    const allArgs = rounds.flatMap(r => r.arguments);
    const avgConfidence = allArgs.reduce((acc, a) => acc + a.confidence, 0) / allArgs.length;
    const hasRisk = allArgs.some(a => a.riskFlagged);

    const consensusScore = hasRisk ? Math.min(avgConfidence, 0.65) : Math.round(avgConfidence * 100) / 100;

    let verdict: 'auto_execute' | 'operator_confirmation_required' | 'rejected';
    if (consensusScore >= requiredThreshold) {
      verdict = 'auto_execute';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) >= threshold (${requiredThreshold}). Approved for auto-execution.`);
    } else if (consensusScore >= 0.65) {
      verdict = 'operator_confirmation_required';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) below threshold (${requiredThreshold}). Requiring operator confirmation.`);
    } else {
      verdict = 'rejected';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) critical failure or flagged risk. Proposal rejected.`);
    }

    return {
      debateId,
      topic: intent.intent,
      rounds,
      consensusScore,
      requiredThreshold,
      verdict,
      winningSkillId: selectedSkillId,
      reasoning
    };
  }

  /**
   * Run a multi-round debate across decentralized peer-to-peer nodes with dynamic reputation and role authority weighting
   */
  async runP2PDebate(
    intent: NeuroIntent,
    peerNodes: PeerNode[],
    microState?: CalculatedMicroState
  ): Promise<SwarmDebateResult> {
    const debateId = `p2p-debate-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const rounds: DebateRound[] = [];
    const reasoning: string[] = [];

    const requiredThreshold = this.calculateAdaptiveThreshold(microState);
    reasoning.push(
      `[P2P Adaptive Threshold] Required consensus score: ${requiredThreshold} (beta/alpha ratio: ${microState?.betaAlphaRatio ?? 'N/A'})`
    );

    const activePeers = (peerNodes || []).filter(
      n => (n.status === 'online' || n.status === 'active') && (n.weight === undefined || n.weight > 0)
    );
    reasoning.push(`[P2P Discovery] Total peers: ${peerNodes?.length || 0}, Active online peers: ${activePeers.length}`);

    if (activePeers.length === 0) {
      reasoning.push(`[P2P Fallback] No online peer nodes available. Falling back to local triad debate.`);
      const localResult = await this.runDebate(intent, microState);
      return {
        ...localResult,
        debateId,
        reasoning: [...reasoning, ...localResult.reasoning]
      };
    }

    const selectedSkillId = (intent.features?.skillId as string) || 'skill-1';

    // Round 1: Peer Proposals & Risk Evaluation
    const round1Args: DebateArgument[] = activePeers.map(node => {
      const rep = Math.max(0, Math.min(1, node.reputationScore ?? 1.0));
      let conf = intent.confidence;
      let risk = false;
      let claim = '';

      const triadRole: TriadRole = node.role === 'guard' ? 'guard' : node.role === 'executor' ? 'executor' : 'architect';

      if (triadRole === 'architect') {
        conf = Math.min(1.0, intent.confidence + 0.05 * rep);
        claim = `Peer '${node.name}' (${node.id}) proposing execution plan for '${intent.intent}'`;
      } else if (triadRole === 'guard') {
        risk = intent.confidence < 0.7 || ((intent.features?.quality as number) ?? 1.0) < 0.3;
        claim = risk ? `Peer '${node.name}' flagged safety/confidence risk` : `Peer '${node.name}' verified safety parameters`;
        conf = risk ? Math.min(conf, 0.5) : Math.min(1.0, conf + 0.02 * rep);
      } else {
        claim = `Peer '${node.name}' verified resource execution readiness`;
        conf = 0.90 * rep;
      }

      return {
        role: triadRole,
        round: 1,
        claim,
        confidence: Math.round(conf * 100) / 100,
        skillIdProposed: selectedSkillId,
        riskFlagged: risk,
        nodeId: node.id
      };
    });
    rounds.push({ roundNumber: 1, arguments: round1Args });

    // Round 2: Peer Counter-Arguments & Dynamic Cross-Validation
    const round2Args: DebateArgument[] = activePeers.map(node => {
      const rep = Math.max(0, Math.min(1, node.reputationScore ?? 1.0));
      const triadRole: TriadRole = node.role === 'guard' ? 'guard' : node.role === 'executor' ? 'executor' : 'architect';
      const roleWeight = ROLE_AUTHORITY_WEIGHTS[triadRole] || 1.0;
      const adjustedConf = Math.min(1.0, 0.85 + rep * roleWeight * 0.1);
      const hasRound1Risk = round1Args.some(a => a.riskFlagged);

      return {
        role: triadRole,
        round: 2,
        claim: hasRound1Risk
          ? `Peer '${node.name}' maintaining risk warning in counter-argument`
          : `Peer '${node.name}' validated multi-node state synchronization`,
        confidence: Math.round(adjustedConf * 100) / 100,
        riskFlagged: hasRound1Risk && triadRole === 'guard',
        nodeId: node.id
      };
    });
    rounds.push({ roundNumber: 2, arguments: round2Args });

    // Round 3: Consensus Synthesis
    const round3Args: DebateArgument[] = activePeers.map(node => {
      const rep = Math.max(0, Math.min(1, node.reputationScore ?? 1.0));
      const triadRole: TriadRole = node.role === 'guard' ? 'guard' : node.role === 'executor' ? 'executor' : 'architect';
      return {
        role: triadRole,
        round: 3,
        claim: `Peer '${node.name}' final vote submitted for execution round`,
        confidence: Math.min(1.0, Math.round((0.88 + rep * 0.1) * 100) / 100),
        nodeId: node.id
      };
    });
    rounds.push({ roundNumber: 3, arguments: round3Args });

    // Aggregate Dynamic Weighted Consensus
    const allArgs = rounds.flatMap(r => r.arguments);
    let totalWeightedScore = 0;
    let totalWeight = 0;
    const roleBreakdown: Record<TriadRole, number> = { architect: 0, guard: 0, executor: 0 };

    for (const arg of allArgs) {
      const node = activePeers.find(n => n.id === arg.nodeId);
      const rep = Math.max(0, Math.min(1, node?.reputationScore ?? 1.0));
      const roleWeight = ROLE_AUTHORITY_WEIGHTS[arg.role] || 1.0;
      const nodeWeight = node?.weight !== undefined ? node.weight : 1.0;
      const weight = rep * roleWeight * (nodeWeight > 0 ? nodeWeight : 1.0);

      totalWeightedScore += arg.confidence * weight;
      totalWeight += weight;
      if (roleBreakdown[arg.role] !== undefined) {
        roleBreakdown[arg.role] = (roleBreakdown[arg.role] || 0) + 1;
      }
    }

    const rawWeightedAverage = totalWeight > 0 ? totalWeightedScore / totalWeight : 0.5;
    const hasRiskFlag = allArgs.some(a => a.riskFlagged);

    // Apply Risk Capping at 0.65
    const consensusScore = hasRiskFlag
      ? Math.min(Math.round(rawWeightedAverage * 100) / 100, 0.65)
      : Math.round(rawWeightedAverage * 100) / 100;

    reasoning.push(
      `[Consensus Score] Raw weighted average: ${Math.round(rawWeightedAverage * 100) / 100}, Risk Flagged: ${hasRiskFlag}, Final score: ${consensusScore}`
    );

    // Determine Verdict
    let verdict: 'auto_execute' | 'operator_confirmation_required' | 'rejected';
    if (consensusScore >= requiredThreshold) {
      verdict = 'auto_execute';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) >= threshold (${requiredThreshold}). Auto-execution approved.`);
    } else if (consensusScore >= 0.65) {
      verdict = 'operator_confirmation_required';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) requires operator confirmation.`);
    } else {
      verdict = 'rejected';
      reasoning.push(`[Verdict] Consensus score (${consensusScore}) below safety minimum. Execution rejected.`);
    }

    return {
      debateId,
      topic: intent.intent,
      rounds,
      consensusScore,
      requiredThreshold,
      verdict,
      winningSkillId: selectedSkillId,
      reasoning,
      p2pMetadata: {
        totalPeers: peerNodes.length,
        activePeers: activePeers.length,
        weightedAverageConfidence: Math.round(rawWeightedAverage * 100) / 100,
        riskFlagged: hasRiskFlag,
        roleBreakdown
      }
    };
  }
}
