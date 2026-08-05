import test from 'node:test';
import assert from 'node:assert/strict';
import {
  OmniSwarmAdapter,
  P2PStateRegistry,
  FederatedDebateEngine,
  ROLE_AUTHORITY_WEIGHTS,
  type PeerNode,
  type SwarmState,
  type NeuroIntent
} from '../lib/neurocore-swarm.ts';

test('P2PStateRegistry - Node registration, reputation clamping, and dynamic weighting', () => {
  const registry = new P2PStateRegistry();

  const nodeActive: PeerNode = {
    id: 'node-1',
    name: 'Architect-Node-1',
    role: 'architect',
    reputationScore: 0.9,
    status: 'active',
    lastSeen: Date.now()
  };

  const nodeSyncing: PeerNode = {
    id: 'node-2',
    name: 'Guard-Node-1',
    role: 'guard',
    reputationScore: 0.8,
    status: 'syncing',
    lastSeen: Date.now()
  };

  const nodeDegraded: PeerNode = {
    id: 'node-3',
    name: 'Executor-Node-1',
    role: 'executor',
    reputationScore: 0.8,
    status: 'degraded',
    lastSeen: Date.now()
  };

  const nodeUntrusted: PeerNode = {
    id: 'node-4',
    name: 'Malicious-Node',
    role: 'executor',
    reputationScore: 0.1, // Below 0.2 threshold
    status: 'active',
    lastSeen: Date.now()
  };

  registry.registerNode(nodeActive);
  registry.registerNode(nodeSyncing);
  registry.registerNode(nodeDegraded);
  registry.registerNode(nodeUntrusted);

  const registeredNodes = registry.getNodes();
  assert.equal(registeredNodes.length, 4);

  // Check dynamic weights
  const retrievedActive = registry.getNode('node-1');
  assert.equal(retrievedActive?.weight, 0.9); // 0.9 * 1.0

  const retrievedSyncing = registry.getNode('node-2');
  assert.equal(retrievedSyncing?.weight, 0.4); // 0.8 * 0.5

  const retrievedDegraded = registry.getNode('node-3');
  assert.equal(retrievedDegraded?.weight, 0.2); // 0.8 * 0.25

  const retrievedUntrusted = registry.getNode('node-4');
  assert.equal(retrievedUntrusted?.weight, 0.0); // reputation < 0.2 -> weight 0
});

test('P2PStateRegistry - Heartbeat recovery and inactive node pruning', () => {
  const registry = new P2PStateRegistry();
  const oldTimestamp = Date.now() - 40000; // 40 seconds ago (stale)

  registry.registerNode({
    id: 'node-stale',
    name: 'Stale-Node',
    role: 'architect',
    reputationScore: 0.85,
    status: 'active',
    lastSeen: oldTimestamp
  });

  registry.registerNode({
    id: 'node-syncing',
    name: 'Syncing-Node',
    role: 'guard',
    reputationScore: 0.9,
    status: 'syncing',
    lastSeen: Date.now()
  });

  // Test Heartbeat
  registry.heartbeat('node-syncing');
  const syncingRecovered = registry.getNode('node-syncing');
  assert.equal(syncingRecovered?.status, 'active');
  assert.equal(syncingRecovered?.weight, 0.9);

  // Test Pruning
  const prunedIds = registry.pruneInactiveNodes(30000);
  assert.deepEqual(prunedIds, ['node-stale']);

  const staleNode = registry.getNode('node-stale');
  assert.equal(staleNode?.status, 'disconnected');
  assert.equal(staleNode?.weight, 0.0);
});

test('P2PStateRegistry - State synchronization and version enforcement', () => {
  const registry = new P2PStateRegistry();

  registry.registerNode({
    id: 'node-valid',
    name: 'Valid-Node',
    role: 'architect',
    reputationScore: 0.95,
    status: 'active',
    lastSeen: Date.now()
  });

  registry.registerNode({
    id: 'node-untrusted',
    name: 'Untrusted-Node',
    role: 'executor',
    reputationScore: 0.1,
    status: 'active',
    lastSeen: Date.now()
  });

  const validProposedState: SwarmState = {
    version: 2,
    timestamp: Date.now(),
    activeIntentsCount: 1,
    consensusThreshold: 0.85,
    globalMetrics: {
      totalActionsExecuted: 10,
      averageConfidence: 0.95,
      activePeersCount: 1
    }
  };

  // Reject state from untrusted source node
  const untrustedSyncResult = registry.syncState(validProposedState, 'node-untrusted');
  assert.equal(untrustedSyncResult.accepted, false);
  assert.match(untrustedSyncResult.reason || '', /zero dynamic weight/);

  // Accept valid state from trusted source node
  const validSyncResult = registry.syncState(validProposedState, 'node-valid');
  assert.equal(validSyncResult.accepted, true);
  assert.equal(validSyncResult.currentConsensusVersion, 2);

  // Reject stale version state proposal
  const staleVersionState: SwarmState = {
    version: 1,
    timestamp: Date.now(),
    activeIntentsCount: 0,
    consensusThreshold: 0.85,
    globalMetrics: {
      totalActionsExecuted: 5,
      averageConfidence: 0.8,
      activePeersCount: 1
    }
  };

  const staleSyncResult = registry.syncState(staleVersionState, 'node-valid');
  assert.equal(staleSyncResult.accepted, false);
  assert.match(staleSyncResult.reason || '', /older than current consensus version/);
});

test('FederatedDebateEngine - P2P debate with role authority weighting & risk capping', async () => {
  const engine = new FederatedDebateEngine();

  const peerNodes: PeerNode[] = [
    {
      id: 'node-arch',
      name: 'Peer Architect',
      role: 'architect',
      reputationScore: 0.9,
      status: 'active',
      lastSeen: Date.now()
    },
    {
      id: 'node-guard',
      name: 'Peer Guard',
      role: 'guard',
      reputationScore: 0.95,
      status: 'active',
      lastSeen: Date.now()
    },
    {
      id: 'node-exec',
      name: 'Peer Executor',
      role: 'executor',
      reputationScore: 0.85,
      status: 'active',
      lastSeen: Date.now()
    }
  ];

  // 1. High confidence intent without risk
  const highConfIntent: NeuroIntent = {
    id: 'intent-p2p-1',
    source: 'eeg',
    intent: 'execute_path_routing',
    confidence: 0.95,
    features: { quality: 0.95 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const result1 = await engine.runP2PDebate(highConfIntent, peerNodes, { betaAlphaRatio: 1.9, alphaPower: 1.0, quality: 0.95 });
  assert.equal(result1.verdict, 'auto_execute');
  assert.equal(result1.requiredThreshold, 0.75); // Beta/alpha ratio >= 1.8 -> threshold 0.75
  assert.equal(result1.p2pMetadata?.activePeers, 3);
  assert.equal(result1.p2pMetadata?.riskFlagged, false);

  // 2. Low confidence intent triggering guard risk flag and capping score at 0.65
  const lowConfIntent: NeuroIntent = {
    id: 'intent-p2p-2',
    source: 'eeg',
    intent: 'execute_path_routing',
    confidence: 0.60, // Triggers risk
    features: { quality: 0.95 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const result2 = await engine.runP2PDebate(lowConfIntent, peerNodes, { betaAlphaRatio: 1.2, alphaPower: 1.0, quality: 0.95 });
  assert.equal(result2.consensusScore, 0.65); // Capped at 0.65 due to risk flag
  assert.equal(result2.verdict, 'operator_confirmation_required');
  assert.equal(result2.p2pMetadata?.riskFlagged, true);

  // 3. Zero active peers fallback to local debate
  const fallbackResult = await engine.runP2PDebate(highConfIntent, []);
  assert.match(fallbackResult.reasoning.join('\n'), /Falling back to local triad debate/);
});

test('OmniSwarmAdapter - P2P peer registration, consensus execution, and emergencyStop cascade', async () => {
  const adapter = new OmniSwarmAdapter();
  await adapter.connect();

  const peer1: PeerNode = {
    id: 'node-adapter-1',
    name: 'Adapter Peer Architect',
    role: 'architect',
    reputationScore: 0.95,
    status: 'active',
    lastSeen: Date.now()
  };

  adapter.registerPeerNode(peer1);
  const peers = adapter.getPeerNodes();
  assert.equal(peers.length, 1);
  assert.equal(peers[0].id, 'node-adapter-1');

  const intent: NeuroIntent = {
    id: 'intent-p2p-adapter',
    source: 'eeg',
    intent: 'swarm_navigation',
    confidence: 0.92,
    features: { skillId: 'skill-2' },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const consensusRes = await adapter.executeP2PConsensus(intent);
  assert.equal(consensusRes.debateResult.verdict, 'auto_execute');
  assert.ok(consensusRes.actionId);

  // Test Emergency Stop 3-stage Cascade
  const stopRes = await adapter.emergencyStop();
  assert.equal(stopRes.stopped, true);

  // Verify P2P network peers were updated to disconnected via cascade
  const updatedPeers = adapter.getPeerNodes();
  assert.equal(updatedPeers[0].status, 'disconnected');
  assert.equal(updatedPeers[0].weight, 0);
});
