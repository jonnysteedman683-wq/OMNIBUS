/**
 * OmniSwarmAdapter — Neurocore SwarmProvider Implementation for OMNIBUS
 * Integrates Neurocore neural control plane with SwarmRuntimeEngine
 */

// @ts-ignore
import swarmRuntime from '../swarm_runtime.js';

const SwarmRuntimeEngineClass =
  (swarmRuntime && swarmRuntime.SwarmRuntimeEngine) ||
  (typeof require !== 'undefined' ? require('../swarm_runtime.js').SwarmRuntimeEngine : null);

export type NeuroIntentSource = "egeg" | "mock" | "audio" | "bci" | "eeg";

export interface NeuroIntent {
  id: string;
  source: NeuroIntentSource;
  intent: string;
  confidence: number;
  features: Record<string, any>;
  timestamp: number;
  requiresConfirmation: boolean;
}

export type SwarmActionStatus = "pending" | "in_progress" | "completed" | "failed" | "rejected";

export interface SwarmAction {
  id: string;
  intentId: string;
  action: string;
  parameters: Record<string, any>;
  status: SwarmActionStatus;
  requiresOperator: boolean;
  estimatedCostUSD?: number;
  log: string[];
  result?: Record<string, any>;
}

export interface SwarmCapabilities {
  roles: string[];
  phases: string[];
  supportsCancel: boolean;
  hardwareControl: boolean;
  maxConcurrency: number;
}

export interface EmergencyStopResult {
  stopped: boolean;
  componentsAffected: string[];
  stateSnapshot: "preserved" | "partial" | "lost";
  recoveryPath: string;
}

export interface SwarmProvider {
  connect(config?: Record<string, any>): Promise<void>;
  capabilities(): Promise<SwarmCapabilities>;
  start(intent: NeuroIntent): Promise<SwarmAction>;
  status(actionId: string): Promise<SwarmAction | null>;
  stop(actionId: string): Promise<boolean>;
  emergencyStop(): Promise<EmergencyStopResult>;
}

export class OmniSwarmAdapter implements SwarmProvider {
  private engine: any;
  private isConnected: boolean = false;
  private pendingActions: Map<string, SwarmAction> = new Map();
  private processedActions: Map<string, SwarmAction> = new Map();
  private actionIdToIntentId: Map<string, string> = new Map();

  constructor(engineInstance?: any) {
    this.engine = engineInstance || null;
  }

  async connect(config?: Record<string, any>): Promise<void> {
    try {
      if (!this.engine) {
        if (SwarmRuntimeEngineClass) {
          this.engine = new SwarmRuntimeEngineClass();
        } else {
          console.warn('[OmniSwarmAdapter] SwarmRuntimeEngine not loaded. Initializing fallback inline engine.');
          this.engine = {
            actionQueue: [],
            enqueueAction: (type: string, target: string, parameters: any = {}, reasoning: string = '') => {
              const action = {
                id: `action-${Date.now()}-${Math.floor(Math.random() * 10000)}`,
                timestamp: Date.now(),
                type,
                target,
                parameters,
                reasoning,
                status: 'pending' as SwarmActionStatus
              };
              this.engine.actionQueue.push(action);
              return action;
            },
            processQueue: async (callback?: (act: any) => void) => {
              const processed: any[] = [];
              while (this.engine.actionQueue.length > 0) {
                const act = this.engine.actionQueue.shift();
                act.status = 'completed';
                processed.push(act);
                if (callback) callback(act);
              }
              return processed;
            }
          };
        }
      }
      this.isConnected = true;
      console.log('[OmniSwarmAdapter] Connected to SwarmRuntimeEngine.', config || {});
    } catch (error) {
      console.error('[OmniSwarmAdapter] Connection error:', error);
      throw error;
    }
  }

  async capabilities(): Promise<SwarmCapabilities> {
    return {
      roles: ['coordinator', 'executor', 'observer', 'debate-agent'],
      phases: ['planning', 'execution', 'verification', 'debate'],
      supportsCancel: true,
      hardwareControl: false,
      maxConcurrency: 10
    };
  }

  async start(intent: NeuroIntent): Promise<SwarmAction> {
    if (!this.isConnected || !this.engine) {
      await this.connect();
    }

    const reasoning = `Decoded intent '${intent.intent}' from source '${intent.source}' (confidence: ${intent.confidence})`;
    const engineAction = this.engine.enqueueAction(
      intent.intent,
      intent.source,
      intent.features || {},
      reasoning
    );

    const swarmAction: SwarmAction = {
      id: engineAction.id,
      intentId: intent.id,
      action: intent.intent,
      parameters: intent.features || {},
      status: (engineAction.status as SwarmActionStatus) || 'pending',
      requiresOperator: intent.requiresConfirmation,
      log: [`[${new Date().toISOString()}] Action enqueued with ID: ${engineAction.id}`]
    };

    this.pendingActions.set(swarmAction.id, swarmAction);
    this.actionIdToIntentId.set(swarmAction.id, intent.id);

    // Trigger async processing of queue
    this.runQueueProcessing();

    return swarmAction;
  }

  private async runQueueProcessing(): Promise<void> {
    if (!this.engine || typeof this.engine.processQueue !== 'function') return;

    try {
      const processed = await this.engine.processQueue((engineAct: any) => {
        const action = this.pendingActions.get(engineAct.id);
        if (action) {
          action.status = (engineAct.status as SwarmActionStatus) || 'completed';
          action.log.push(`[${new Date().toISOString()}] Status updated to: ${action.status}`);
          if (action.status === 'completed') {
            action.result = { completedAt: Date.now(), engineActionId: engineAct.id };
          }
          this.processedActions.set(action.id, action);
          this.pendingActions.delete(action.id);
        }
      });

      if (Array.isArray(processed)) {
        for (const engineAct of processed) {
          const action = this.pendingActions.get(engineAct.id) || this.processedActions.get(engineAct.id);
          if (action) {
            action.status = (engineAct.status as SwarmActionStatus) || 'completed';
            if (!this.processedActions.has(action.id)) {
              this.processedActions.set(action.id, action);
              this.pendingActions.delete(action.id);
            }
          }
        }
      }
    } catch (error) {
      console.error('[OmniSwarmAdapter] Queue processing error:', error);
    }
  }

  async status(actionId: string): Promise<SwarmAction | null> {
    if (this.pendingActions.has(actionId)) {
      return this.pendingActions.get(actionId) || null;
    }
    if (this.processedActions.has(actionId)) {
      return this.processedActions.get(actionId) || null;
    }

    if (this.engine && Array.isArray(this.engine.actionQueue)) {
      const foundInQueue = this.engine.actionQueue.find((a: any) => a.id === actionId);
      if (foundInQueue) {
        return {
          id: foundInQueue.id,
          intentId: this.actionIdToIntentId.get(actionId) || 'unknown',
          action: foundInQueue.type,
          parameters: foundInQueue.parameters,
          status: (foundInQueue.status as SwarmActionStatus) || 'pending',
          requiresOperator: false,
          log: [`[${new Date().toISOString()}] Action queried from engine queue`]
        };
      }
    }

    return null;
  }

  async stop(actionId: string): Promise<boolean> {
    if (this.pendingActions.has(actionId)) {
      const action = this.pendingActions.get(actionId)!;
      action.status = 'rejected';
      action.log.push(`[${new Date().toISOString()}] Action manually stopped.`);
      this.processedActions.set(actionId, action);
      this.pendingActions.delete(actionId);

      if (this.engine && Array.isArray(this.engine.actionQueue)) {
        this.engine.actionQueue = this.engine.actionQueue.filter((a: any) => a.id !== actionId);
      }
      return true;
    }
    return false;
  }

  async emergencyStop(): Promise<EmergencyStopResult> {
    const affected: string[] = [];

    for (const [id, action] of this.pendingActions.entries()) {
      action.status = 'rejected';
      action.log.push(`[${new Date().toISOString()}] Emergency stop triggered.`);
      this.processedActions.set(id, action);
      affected.push(id);
    }
    this.pendingActions.clear();

    if (this.engine && Array.isArray(this.engine.actionQueue)) {
      this.engine.actionQueue = [];
    }

    console.warn(`[OmniSwarmAdapter] Emergency stop executed. Cleared ${affected.length} pending action(s).`);

    return {
      stopped: true,
      componentsAffected: ['SwarmRuntimeEngine', 'ActionQueue', ...affected],
      stateSnapshot: 'preserved',
      recoveryPath: 'Re-connect adapter and re-submit intents.'
    };
  }
}
