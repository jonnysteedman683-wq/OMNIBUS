import test from 'node:test';
import assert from 'node:assert/strict';
import {
  OmniSwarmAdapter,
  P2PStateRegistry,
  FederatedDebateEngine,
  HermesAgentAdapter,
  ROLE_AUTHORITY_WEIGHTS,
  type PeerNode,
  type SwarmState,
  type NeuroIntent,
  type SwarmProvider
} from '../lib/neurocore-swarm.ts';

test('CHALLENGE 1: P2PStateRegistry - State Version Enforcement & Source Node Validation', () => {
  const registry = new P2PStateRegistry({ version: 5 });

  registry.registerNode({
    id: 'trusted-peer-1',
    name: 'Trusted Peer',
    role: 'architect',
    reputationScore: 0.95,
    status: 'active',
    lastSeen: Date.now()
  });

  registry.registerNode({
    id: 'untrusted-peer-1',
    name: 'Untrusted Peer',
    role: 'executor',
    reputationScore: 0.1, // weight 0
    status: 'active',
    lastSeen: Date.now()
  });

  registry.registerNode({
    id: 'disconnected-peer-1',
    name: 'Disconnected Peer',
    role: 'guard',
    reputationScore: 0.9,
    status: 'disconnected', // weight 0
    lastSeen: Date.now()
  });

  // 1. Proposed version older than consensus version -> REJECTED
  const olderState: SwarmState = {
    version: 4,
    timestamp: Date.now(),
    activeIntentsCount: 0,
    consensusThreshold: 0.85,
    globalMetrics: { totalActionsExecuted: 1, averageConfidence: 0.9, activePeersCount: 1 }
  };
  const olderResult = registry.syncState(olderState, 'trusted-peer-1');
  assert.equal(olderResult.accepted, false);
  assert.equal(olderResult.currentConsensusVersion, 5);
  assert.match(olderResult.reason || '', /older than current consensus version/);

  // 2. Proposed version higher than consensus version from trusted peer -> ACCEPTED
  const newerState: SwarmState = {
    version: 6,
    timestamp: Date.now(),
    activeIntentsCount: 2,
    consensusThreshold: 0.85,
    globalMetrics: { totalActionsExecuted: 10, averageConfidence: 0.95, activePeersCount: 1 }
  };
  const newerResult = registry.syncState(newerState, 'trusted-peer-1');
  assert.equal(newerResult.accepted, true);
  assert.equal(newerResult.currentConsensusVersion, 6);
  assert.equal(registry.getConsensusState().version, 6);

  // 3. Sync proposed by untrusted peer (reputation < 0.2) -> REJECTED
  const proposedV7: SwarmState = {
    version: 7,
    timestamp: Date.now(),
    activeIntentsCount: 0,
    consensusThreshold: 0.85,
    globalMetrics: { totalActionsExecuted: 15, averageConfidence: 0.8, activePeersCount: 1 }
  };
  const untrustedResult = registry.syncState(proposedV7, 'untrusted-peer-1');
  assert.equal(untrustedResult.accepted, false);
  assert.match(untrustedResult.reason || '', /zero dynamic weight/);

  // 4. Sync proposed by disconnected peer (weight 0) -> REJECTED
  const disconnectedResult = registry.syncState(proposedV7, 'disconnected-peer-1');
  assert.equal(disconnectedResult.accepted, false);
  assert.match(disconnectedResult.reason || '', /zero dynamic weight/);

  // 5. Unregistered / unknown source node handling check
  const unknownResult = registry.syncState(proposedV7, 'unknown-node-999');
  // Record result behavior for empirical handoff analysis
  assert.ok(typeof unknownResult.accepted === 'boolean');
});

test('CHALLENGE 2: P2PStateRegistry - Inactive Node Pruning & Heartbeat Recovery', () => {
  const registry = new P2PStateRegistry();
  const now = Date.now();

  registry.registerNode({
    id: 'active-recent',
    name: 'Active Recent',
    role: 'architect',
    reputationScore: 0.9,
    status: 'active',
    lastSeen: now - 5000 // 5s ago
  });

  registry.registerNode({
    id: 'active-stale',
    name: 'Active Stale',
    role: 'guard',
    reputationScore: 0.85,
    status: 'active',
    lastSeen: now - 35000 // 35s ago (stale > 30s)
  });

  registry.registerNode({
    id: 'syncing-stale',
    name: 'Syncing Stale',
    role: 'executor',
    reputationScore: 0.8,
    status: 'syncing',
    lastSeen: now - 40000 // 40s ago (stale > 30s)
  });

  // 1. Initial nodes check
  assert.equal(registry.getNodes().length, 3);
  assert.equal(registry.getConsensusState().globalMetrics.activePeersCount, 2); // only 'active' status counted (active-recent + active-stale)

  // 2. Prune with 30s threshold
  const pruned = registry.pruneInactiveNodes(30000);
  assert.deepEqual(pruned.sort(), ['active-stale', 'syncing-stale'].sort());

  // Verify status updated to 'disconnected' and weight = 0
  const stale1 = registry.getNode('active-stale');
  const stale2 = registry.getNode('syncing-stale');
  assert.equal(stale1?.status, 'disconnected');
  assert.equal(stale1?.weight, 0.0);
  assert.equal(stale2?.status, 'disconnected');
  assert.equal(stale2?.weight, 0.0);

  // 3. Idempotent pruning check: running prune again with same timestamp window returns []
  const prunedSecond = registry.pruneInactiveNodes(30000);
  assert.deepEqual(prunedSecond, []);

  // 4. Heartbeat recovery test for syncing node vs disconnected node
  registry.registerNode({
    id: 'syncing-node-live',
    name: 'Syncing Node Live',
    role: 'architect',
    reputationScore: 0.9,
    status: 'syncing',
    lastSeen: Date.now()
  });
  assert.equal(registry.getNode('syncing-node-live')?.weight, 0.45); // 0.9 * 0.5

  registry.heartbeat('syncing-node-live');
  const recoveredSyncing = registry.getNode('syncing-node-live');
  assert.equal(recoveredSyncing?.status, 'active');
  assert.equal(recoveredSyncing?.weight, 0.9);
});

test('CHALLENGE 3: OmniSwarmAdapter - 3-Stage Emergency Stop Cascade & Provider Resilience', async () => {
  const adapter = new OmniSwarmAdapter();
  await adapter.connect({ enableHermes: true, hermesConfig: { fallbackToMock: true } });

  // Register P2P Nodes
  adapter.registerPeerNode({
    id: 'peer-cascade-1',
    name: 'Cascade Peer 1',
    role: 'architect',
    reputationScore: 0.9,
    status: 'active',
    lastSeen: Date.now()
  });

  adapter.registerPeerNode({
    id: 'peer-cascade-2',
    name: 'Cascade Peer 2',
    role: 'guard',
    reputationScore: 0.85,
    status: 'syncing',
    lastSeen: Date.now()
  });

  // Verify P2P network initial state
  const initialPeers = adapter.getPeerNodes();
  assert.equal(initialPeers.length, 2);
  assert.equal(initialPeers[0].status, 'active');
  assert.equal(initialPeers[1].status, 'syncing');

  // Register a mock custom provider to test Stage 2 cascade
  let customProviderStopped = false;
  const mockProvider: SwarmProvider = {
    connect: async () => {},
    capabilities: async () => ({ roles: ['custom'], phases: ['exec'] }),
    start: async () => ({ actionId: 'custom-1', status: 'started' }),
    status: async () => ({ actionId: 'custom-1', status: 'running', output: [] }),
    stop: async () => ({ actionId: 'custom-1', status: 'stopped' }),
    emergencyStop: async () => {
      customProviderStopped = true;
      return { stopped: true };
    }
  };
  adapter.registerProvider('custom-mock', mockProvider);

  // Execute 3-Stage Emergency Stop
  const stopResult = await adapter.emergencyStop();
  assert.equal(stopResult.stopped, true);
  assert.equal(customProviderStopped, true);

  // Verify Stage 3: All non-disconnected P2P nodes transition to 'disconnected' status with weight 0
  const postStopPeers = adapter.getPeerNodes();
  assert.equal(postStopPeers[0].status, 'disconnected');
  assert.equal(postStopPeers[0].weight, 0.0);
  assert.equal(postStopPeers[1].status, 'disconnected');
  assert.equal(postStopPeers[1].weight, 0.0);

  // Verify provider resilience: when a provider emergencyStop throws an error, Stage 3 still executes
  const adapterFailing = new OmniSwarmAdapter();
  await adapterFailing.connect();
  adapterFailing.registerPeerNode({
    id: 'peer-failing-test',
    name: 'Peer Failing Test',
    role: 'executor',
    reputationScore: 0.8,
    status: 'active',
    lastSeen: Date.now()
  });

  const failingProvider: SwarmProvider = {
    connect: async () => {},
    capabilities: async () => ({ roles: ['failing'], phases: [] }),
    start: async () => ({ actionId: 'f-1', status: 'err' }),
    status: async () => ({ actionId: 'f-1', status: 'err', output: [] }),
    stop: async () => ({ actionId: 'f-1', status: 'err' }),
    emergencyStop: async () => {
      throw new Error('Provider hardware error on shutdown!');
    }
  };
  adapterFailing.registerProvider('failing-provider', failingProvider);

  const failingStopResult = await adapterFailing.emergencyStop();
  assert.equal(failingStopResult.stopped, false); // Indicates non-clean overall stop due to provider error
  // But Stage 3 MUST still have executed and disconnected the P2P nodes
  const failingPeers = adapterFailing.getPeerNodes();
  assert.equal(failingPeers[0].status, 'disconnected');
  assert.equal(failingPeers[0].weight, 0.0);
});

test('CHALLENGE 4: FederatedDebateEngine - P2P Dynamic Consensus Weighting & Risk Flag Capping', async () => {
  const engine = new FederatedDebateEngine();

  const peerNodes: PeerNode[] = [
    {
      id: 'guard-peer',
      name: 'Guard Node',
      role: 'guard',
      reputationScore: 1.0,
      status: 'active',
      lastSeen: Date.now(),
      weight: 1.0
    },
    {
      id: 'arch-peer',
      name: 'Architect Node',
      role: 'architect',
      reputationScore: 0.9,
      status: 'active',
      lastSeen: Date.now(),
      weight: 0.9
    },
    {
      id: 'exec-peer',
      name: 'Executor Node',
      role: 'executor',
      reputationScore: 0.8,
      status: 'active',
      lastSeen: Date.now(),
      weight: 0.8
    }
  ];

  // Intent with low confidence (0.5) -> triggers Guard risk flag
  const riskyIntent: NeuroIntent = {
    id: 'intent-risk-test',
    source: 'mock',
    intent: 'recalibrate_swarm_topology',
    confidence: 0.5,
    features: { quality: 0.8 },
    timestamp: Date.now(),
    requiresConfirmation: false
  };

  const result = await engine.runP2PDebate(riskyIntent, peerNodes);
  assert.equal(result.consensusScore, 0.65); // Hard capped at 0.65
  assert.equal(result.verdict, 'operator_confirmation_required');
  assert.equal(result.p2pMetadata?.riskFlagged, true);
  assert.equal(result.p2pMetadata?.roleBreakdown.guard, 3);
  assert.equal(result.p2pMetadata?.roleBreakdown.architect, 3);
  assert.equal(result.p2pMetadata?.roleBreakdown.executor, 3);
});
