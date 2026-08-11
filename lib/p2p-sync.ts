import { type CalculatedMicroState } from './neurostream-adapter.ts';
import { type TriadRole } from './swarm-debate.ts';

/** Operational status of a peer node in the swarm */
export type NodeStatus = 'active' | 'degraded' | 'disconnected' | 'syncing';

/** Data contract representing a peer node in the P2P swarm mesh */
export interface PeerNode {
  id: string;
  name: string;
  role: TriadRole | 'coordinator' | 'observer' | string;
  reputationScore: number; // Normalized range: 0.0 (untrusted) to 1.0 (fully trusted)
  status: NodeStatus;
  lastSeen: number; // Unix timestamp in milliseconds
  weight?: number; // Computed dynamic weight: f(reputationScore, status)
  endpoint?: string;
  metadata?: Record<string, unknown>;
}

/** Data contract for collective swarm state snapshot */
export interface SwarmState {
  version: number;
  timestamp: number;
  leaderId?: string;
  activeIntentsCount: number;
  consensusThreshold: number;
  globalMetrics: {
    totalActionsExecuted: number;
    averageConfidence: number;
    activePeersCount: number;
  };
  microStateSnapshot?: CalculatedMicroState;
  customData?: Record<string, unknown>;
}

/** Result object returned following a state synchronization proposal */
export interface SwarmSyncResult {
  syncId: string;
  accepted: boolean;
  proposedVersion: number;
  currentConsensusVersion: number;
  participatingPeersCount: number;
  weightedAgreementRatio: number;
  state: SwarmState;
  timestamp: number;
  reason?: string;
}

export class P2PStateRegistry {
  private nodes: Map<string, PeerNode> = new Map();
  private currentState: SwarmState;
  private syncHistory: SwarmSyncResult[] = [];
  private maxStaleMs: number = 30000; // 30s timeout for node pruning

  constructor(initialState?: Partial<SwarmState>) {
    this.currentState = {
      version: 1,
      timestamp: Date.now(),
      activeIntentsCount: 0,
      consensusThreshold: 0.85,
      globalMetrics: {
        totalActionsExecuted: 0,
        averageConfidence: 1.0,
        activePeersCount: 0
      },
      ...initialState
    };
  }

  // --- Peer Management Methods ---

  registerNode(node: PeerNode): void {
    const validatedReputation = Math.max(0, Math.min(1, node.reputationScore ?? 1.0));
    const fullNode: PeerNode = {
      ...node,
      reputationScore: validatedReputation,
      lastSeen: node.lastSeen || Date.now(),
      status: node.status || 'active'
    };
    fullNode.weight = this.calculateNodeWeight(fullNode);
    this.nodes.set(fullNode.id, fullNode);
    this.updateGlobalMetrics();
  }

  unregisterNode(nodeId: string): boolean {
    const existed = this.nodes.delete(nodeId);
    if (existed) {
      this.updateGlobalMetrics();
    }
    return existed;
  }

  getNodes(): PeerNode[] {
    return Array.from(this.nodes.values()).map(n => ({
      ...n,
      weight: this.calculateNodeWeight(n)
    }));
  }

  getNode(id: string): PeerNode | undefined {
    const node = this.nodes.get(id);
    if (!node) return undefined;
    return {
      ...node,
      weight: this.calculateNodeWeight(node)
    };
  }

  updateNodeReputation(nodeId: string, score: number): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.reputationScore = Math.max(0, Math.min(1, score));
      node.weight = this.calculateNodeWeight(node);
      this.nodes.set(nodeId, node);
    }
  }

  updateNodeStatus(nodeId: string, status: NodeStatus): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.status = status;
      node.weight = this.calculateNodeWeight(node);
      this.nodes.set(nodeId, node);
      this.updateGlobalMetrics();
    }
  }

  heartbeat(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (node) {
      node.lastSeen = Date.now();
      if (node.status === 'syncing' || node.status === 'degraded') {
        node.status = 'active';
      }
      node.weight = this.calculateNodeWeight(node);
      this.nodes.set(nodeId, node);
    }
  }

  pruneInactiveNodes(timeoutMs: number = this.maxStaleMs): string[] {
    const now = Date.now();
    const affectedNodeIds: string[] = [];

    for (const [id, node] of this.nodes.entries()) {
      if (now - node.lastSeen > timeoutMs) {
        if (node.status !== 'disconnected') {
          node.status = 'disconnected';
          node.weight = 0;
          affectedNodeIds.push(id);
        }
      }
    }

    if (affectedNodeIds.length > 0) {
      this.updateGlobalMetrics();
    }

    return affectedNodeIds;
  }

  // --- Dynamic Weight Calculation ---

  calculateNodeWeight(node: PeerNode): number {
    if (node.status === 'disconnected') return 0.0;
    if (node.reputationScore < 0.2) return 0.0; // Untrusted peer cutoff

    let statusMultiplier = 1.0;
    switch (node.status) {
      case 'active':
        statusMultiplier = 1.0;
        break;
      case 'syncing':
        statusMultiplier = 0.5;
        break;
      case 'degraded':
        statusMultiplier = 0.25;
        break;
      default:
        statusMultiplier = 0.0;
    }

    const weight = Math.round(node.reputationScore * statusMultiplier * 100) / 100;
    return weight;
  }

  // --- Collective State Synchronization ---

  syncState(proposedState: SwarmState, sourceNodeId?: string): SwarmSyncResult {
    const syncId = `sync-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const now = Date.now();

    // Verify source node reputation if provided
    if (sourceNodeId) {
      const sourceNode = this.getNode(sourceNodeId);
      if (sourceNode && sourceNode.weight === 0) {
        return {
          syncId,
          accepted: false,
          proposedVersion: proposedState.version,
          currentConsensusVersion: this.currentState.version,
          participatingPeersCount: this.getNodes().filter(n => (n.weight ?? 0) > 0).length,
          weightedAgreementRatio: 0,
          state: this.currentState,
          timestamp: now,
          reason: `Source node '${sourceNodeId}' has zero dynamic weight (status=${sourceNode.status}, reputation=${sourceNode.reputationScore}).`
        };
      }
    }

    // Version rule: State update must advance version or match with valid newer timestamp
    if (proposedState.version < this.currentState.version) {
      return {
        syncId,
        accepted: false,
        proposedVersion: proposedState.version,
        currentConsensusVersion: this.currentState.version,
        participatingPeersCount: this.getNodes().filter(n => (n.weight ?? 0) > 0).length,
        weightedAgreementRatio: 0,
        state: this.currentState,
        timestamp: now,
        reason: `Proposed version (${proposedState.version}) is older than current consensus version (${this.currentState.version}).`
      };
    }

    // Weighted consensus agreement calculation across active peer quorum
    const activePeers = this.getNodes().filter(n => n.status !== 'disconnected' && (n.weight ?? 0) > 0);
    const totalWeight = activePeers.reduce((sum, n) => sum + (n.weight ?? 0), 0);
    
    // If source node is valid and weight total exists, compute agreement ratio
    const sourceWeight = sourceNodeId ? (this.getNode(sourceNodeId)?.weight ?? 1.0) : 1.0;
    const weightedAgreementRatio = totalWeight > 0 ? Math.min(1.0, sourceWeight / totalWeight) : 1.0;

    // Update consensus state
    this.currentState = {
      ...proposedState,
      timestamp: now,
      globalMetrics: {
        ...proposedState.globalMetrics,
        activePeersCount: activePeers.length
      }
    };

    const syncResult: SwarmSyncResult = {
      syncId,
      accepted: true,
      proposedVersion: proposedState.version,
      currentConsensusVersion: this.currentState.version,
      participatingPeersCount: activePeers.length,
      weightedAgreementRatio,
      state: this.currentState,
      timestamp: now
    };

    this.syncHistory.push(syncResult);
    return syncResult;
  }

  getConsensusState(): SwarmState {
    return { ...this.currentState };
  }

  getSyncHistory(): SwarmSyncResult[] {
    return [...this.syncHistory];
  }

  /**
   * Broadcast emergency stop to all registered nodes, setting non-disconnected nodes to disconnected.
   */
  async broadcastEmergencyStop(): Promise<{ success: boolean; haltedNodesCount: number }> {
    let haltedCount = 0;
    for (const [id, node] of this.nodes.entries()) {
      if (node.status !== 'disconnected') {
        node.status = 'disconnected';
        node.weight = 0;
        haltedCount++;
      }
    }
    this.updateGlobalMetrics();
    return { success: true, haltedNodesCount: haltedCount };
  }

  private updateGlobalMetrics(): void {
    const activePeers = Array.from(this.nodes.values()).filter(n => n.status === 'active');
    this.currentState.globalMetrics.activePeersCount = activePeers.length;
  }
}
