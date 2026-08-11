import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SwarmUpgradeRegistry, type SystemHealthReport } from './upgrade-manifest.ts';
import { NeuroStreamAdapter } from './neurostream-adapter.ts';
import { FederatedDebateEngine, type SwarmDebateResult, ROLE_AUTHORITY_WEIGHTS } from './swarm-debate.ts';
import { HermesAgentAdapter, type HermesConfig } from './hermes-adapter.ts';
import { P2PStateRegistry, type PeerNode, type SwarmState, type SwarmSyncResult, type NodeStatus } from './p2p-sync.ts';

export { HermesAgentAdapter, type HermesConfig } from './hermes-adapter.ts';
export { NeuroStreamAdapter } from './neurostream-adapter.ts';
export { SwarmUpgradeRegistry, type SystemHealthReport, type SwarmUpgradeTask } from './upgrade-manifest.ts';
export { FederatedDebateEngine, type SwarmDebateResult, type TriadRole, type DebateRound, type DebateArgument, ROLE_AUTHORITY_WEIGHTS } from './swarm-debate.ts';
export { P2PStateRegistry, type PeerNode, type SwarmState, type SwarmSyncResult, type NodeStatus } from './p2p-sync.ts';

let SwarmRuntimeEngineClass: any = null;

function loadSwarmRuntimeEngine() {
  if (SwarmRuntimeEngineClass) return SwarmRuntimeEngineClass;

  try {
    const currentDir = path.dirname(fileURLToPath(import.meta.url));
    const runtimePath = path.resolve(currentDir, '../swarm_runtime.js');
    const code = fs.readFileSync(runtimePath, 'utf8');

    const customModule = { exports: {} as any };
    const runner = new Function('module', 'exports', code);
    runner(customModule, customModule.exports);

    SwarmRuntimeEngineClass = customModule.exports.SwarmRuntimeEngine;
    return SwarmRuntimeEngineClass;
  } catch (err) {
    return null;
  }
}

export interface NeuroIntent {
  id: string;
  source: 'eeg' | 'mock' | 'audio' | 'bci';
  intent: string;
  confidence: number;
  features: Record<string, unknown>;
  timestamp: number;
  requiresConfirmation: boolean;
}

export interface SwarmProvider {
  connect(config?: { baseUrl?: string; maxConcurrency?: number }): Promise<void>;
  capabilities(): Promise<{ roles: string[]; phases: string[] }>;
  start(intent: NeuroIntent): Promise<{ actionId: string; status: string }>;
  status(actionId: string): Promise<{ actionId: string; status: string; output: string[] }>;
  stop(actionId: string): Promise<{ actionId: string; status: string }>;
  emergencyStop(): Promise<{ stopped: boolean }>;
}

export class OmniSwarmAdapter implements SwarmProvider {
  private engine: any = null;
  private pendingActions: Map<string, any> = new Map();
  private processedActions: Map<string, any> = new Map();
  private upgradeRegistry: SwarmUpgradeRegistry = new SwarmUpgradeRegistry();
  private streamAdapter: NeuroStreamAdapter = new NeuroStreamAdapter();
  private debateEngine: FederatedDebateEngine = new FederatedDebateEngine();
  private p2pRegistry: P2PStateRegistry = new P2PStateRegistry();
  private providers: Map<string, SwarmProvider> = new Map();
  private defaultProvider: string = 'runtime';

  async connect(config: { baseUrl?: string; maxConcurrency?: number; enableHermes?: boolean; hermesConfig?: HermesConfig } = {}): Promise<void> {
    try {
      const EngineClass = loadSwarmRuntimeEngine();

      if (!EngineClass) {
        throw new Error('SwarmRuntimeEngine module could not be loaded from ./swarm_runtime.js');
      }

      this.engine = new EngineClass();
      if (!this.engine) {
        throw new Error('Failed to instantiate SwarmRuntimeEngine instance.');
      }

      this.providers.set('runtime', this);

      if (config.enableHermes || config.hermesConfig) {
        const hermes = new HermesAgentAdapter(config.hermesConfig);
        await hermes.connect(config.hermesConfig);
        this.registerProvider('hermes', hermes);
      }
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.connect] Error connecting to SwarmRuntimeEngine: ${error.message}`);
    }
  }

  registerProvider(name: string, provider: SwarmProvider): void {
    this.providers.set(name, provider);
  }

  getProvider(name: string): SwarmProvider | undefined {
    return this.providers.get(name);
  }

  getProviders(): Map<string, SwarmProvider> {
    return this.providers;
  }

  setDefaultProvider(name: string): void {
    if (!this.providers.has(name) && name !== 'runtime') {
      throw new Error(`Provider '${name}' is not registered.`);
    }
    this.defaultProvider = name;
  }

  getUpgradeRegistry(): SwarmUpgradeRegistry {
    return this.upgradeRegistry;
  }

  getStreamAdapter(): NeuroStreamAdapter {
    return this.streamAdapter;
  }

  getDebateEngine(): FederatedDebateEngine {
    return this.debateEngine;
  }

  getP2PRegistry(): P2PStateRegistry {
    return this.p2pRegistry;
  }

  getP2PStateRegistry(): P2PStateRegistry {
    return this.p2pRegistry;
  }

  registerPeerNode(node: PeerNode): void {
    this.p2pRegistry.registerNode(node);
  }

  getPeerNodes(): PeerNode[] {
    return this.p2pRegistry.getNodes();
  }

  syncSwarmState(state: SwarmState, sourceNodeId?: string): SwarmSyncResult {
    return this.p2pRegistry.syncState(state, sourceNodeId);
  }

  getConsensusState(): SwarmState {
    return this.p2pRegistry.getConsensusState();
  }

  async runDebate(intent: NeuroIntent): Promise<SwarmDebateResult> {
    const microState = this.streamAdapter.analyzeMicroState();
    return this.debateEngine.runDebate(intent, microState);
  }

  async executeP2PConsensus(
    intent: NeuroIntent
  ): Promise<{ debateResult: SwarmDebateResult; actionId?: string; status: string }> {
    const microState = this.streamAdapter.analyzeMicroState();
    const peerNodes = this.p2pRegistry.getNodes();
    const debateResult = await this.debateEngine.runP2PDebate(intent, peerNodes, microState);

    if (debateResult.verdict === 'auto_execute') {
      const startRes = await this.start(intent);
      return {
        debateResult,
        actionId: startRes.actionId,
        status: startRes.status
      };
    } else if (debateResult.verdict === 'operator_confirmation_required') {
      const startRes = await this.start({ ...intent, requiresConfirmation: true });
      return {
        debateResult,
        actionId: startRes.actionId,
        status: startRes.status
      };
    } else {
      return {
        debateResult,
        status: 'rejected'
      };
    }
  }

  async getHealthMetrics(): Promise<SystemHealthReport> {
    return this.upgradeRegistry.runHealthDiagnostics();
  }

  async capabilities(): Promise<{ roles: string[]; phases: string[] }> {
    try {
      if (!this.engine) {
        throw new Error('Adapter not connected. Call connect() first.');
      }

      const skillCards = this.engine.skillCards ? Array.from(this.engine.skillCards.values()) : [];
      const roles = skillCards.length > 0
        ? skillCards.map((s: any) => s.name || s.id)
        : ['coordinator', 'executor', 'observer'];

      return {
        roles: [...roles, 'architect', 'guard'],
        phases: ['planning', 'execution', 'verification', 'debate']
      };
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.capabilities] Error retrieving capabilities: ${error.message}`);
    }
  }

  async start(intent: NeuroIntent): Promise<{ actionId: string; status: string }> {
    try {
      if (!this.engine) {
        throw new Error('Adapter not connected. Call connect() first.');
      }

      const target = `intent-${intent.id}`;
      const parameters = intent.features || {};
      const reasoning = `NeuroIntent from ${intent.source}: ${intent.intent} (confidence=${intent.confidence})`;

      if (typeof this.engine.enqueueAction !== 'function') {
        throw new Error('SwarmRuntimeEngine missing enqueueAction method.');
      }

      const action = this.engine.enqueueAction(
        intent.intent,
        target,
        parameters,
        reasoning
      );

      this.pendingActions.set(action.id, action);

      if (!intent.requiresConfirmation) {
        if (typeof this.engine.processQueue !== 'function') {
          throw new Error('SwarmRuntimeEngine missing processQueue method.');
        }

        const processed = await this.engine.processQueue((completedAction: any) => {
          this.pendingActions.delete(completedAction.id);
          this.processedActions.set(completedAction.id, completedAction);
        });

        const result = Array.isArray(processed)
          ? processed.find((a: any) => a.id === action.id)
          : null;

        return {
          actionId: action.id,
          status: result?.status || 'completed'
        };
      }

      return {
        actionId: action.id,
        status: 'pending_confirmation'
      };
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.start] Error starting intent '${intent?.id}': ${error.message}`);
    }
  }

  async status(actionId: string): Promise<{ actionId: string; status: string; output: string[] }> {
    try {
      if (!this.engine) {
        throw new Error('Adapter not connected. Call connect() first.');
      }

      const action = this.processedActions.get(actionId) || this.pendingActions.get(actionId);

      if (!action) {
        return {
          actionId,
          status: 'unknown',
          output: ['Action not found in engine state.']
        };
      }

      const output: string[] = [];
      if (action.reasoning) output.push(action.reasoning);
      output.push(`Status: ${action.status}`);

      return {
        actionId,
        status: action.status,
        output
      };
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.status] Error querying status for action '${actionId}': ${error.message}`);
    }
  }

  async stop(actionId: string): Promise<{ actionId: string; status: string }> {
    try {
      if (!this.engine) {
        throw new Error('Adapter not connected. Call connect() first.');
      }

      const existed = this.pendingActions.delete(actionId);
      return {
        actionId,
        status: existed ? 'cancelled' : 'not_found'
      };
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.stop] Error stopping action '${actionId}': ${error.message}`);
    }
  }

  async emergencyStop(): Promise<{ stopped: boolean }> {
    try {
      if (!this.engine) {
        throw new Error('Adapter not connected. Call connect() first.');
      }

      // Stage 1: Clear local pending queue
      this.pendingActions.clear();

      let allStopped = true;

      // Stage 2: Cascade stop to registered provider adapters
      for (const [_, provider] of this.providers.entries()) {
        if (provider !== (this as unknown as SwarmProvider)) {
          try {
            const res = await provider.emergencyStop();
            if (!res.stopped) allStopped = false;
          } catch {
            allStopped = false;
          }
        }
      }

      // Stage 3: Broadcast emergency stop to P2P registry network
      try {
        const p2pRes = await this.p2pRegistry.broadcastEmergencyStop();
        if (!p2pRes.success) allStopped = false;
      } catch {
        allStopped = false;
      }

      return { stopped: allStopped };
    } catch (error: any) {
      throw new Error(`[OmniSwarmAdapter.emergencyStop] Error executing emergency stop: ${error.message}`);
    }
  }
}
