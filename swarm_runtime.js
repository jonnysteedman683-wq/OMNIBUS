/**
 * OMNIBUS — Autonomous Swarm & Sidekick Runtime
 * Harvested & Adapted from SUPRIME-SWARM & AI Sidekick
 *
 * Features:
 * - Autonomy Action Queue with preconditions & success rate metrics
 * - Federated Consensus Router & Multi-Agent Swarm Debate
 * - SkillCard Execution Engine
 * - Graph-based workflow execution with loops
 */

class SkillCard {
  constructor(id, name, description, preconditions = {}) {
    this.id = id;
    this.name = name;
    this.description = description;
    this.preconditions = preconditions;
    this.successRate = 1.0;
    this.useCount = 0;
    this.lastUsed = Date.now();
  }

  recordUsage(success = true) {
    this.useCount++;
    this.lastUsed = Date.now();
    this.successRate = (this.successRate * (this.useCount - 1) + (success ? 1 : 0)) / this.useCount;
  }
}

class SwarmAutonomyAction {
  constructor(id, type, target, parameters = {}, reasoning = '') {
    this.id = id || `action-${Date.now()}`;
    this.timestamp = Date.now();
    this.type = type;
    this.target = target;
    this.parameters = parameters;
    this.reasoning = reasoning;
    this.status = 'pending'; // pending, in_progress, completed, failed
  }
}

class GraphNode {
  constructor(id, action, loop = null) {
    this.id = id;
    this.action = action;
    this.loop = loop; // { type: 'for'|'while'|'repeat', count, condition, step }
    this.incoming = new Set();
    this.outgoing = new Set();
    this.visited = false;
    this.output = null;
  }
}

class GraphEdge {
  constructor(from, to, condition = null) {
    this.from = from;
    this.to = to;
    this.condition = condition; // optional guard predicate
  }
}

class SwarmGraph {
  constructor(name = 'default') {
    this.name = name;
    this.nodes = new Map();
    this.edges = [];
    this.entryNodes = new Set();
    this.exitNodes = new Set();
  }

  addNode(id, action, loop = null) {
    const node = new GraphNode(id, action, loop);
    this.nodes.set(id, node);
    if (!node.incoming.size) this.entryNodes.add(id);
    if (!node.outgoing.size) this.exitNodes.add(id);
    return node;
  }

  addEdge(fromId, toId, condition = null) {
    const from = this.nodes.get(fromId);
    const to = this.nodes.get(toId);
    if (!from || !to) return;
    from.outgoing.add(toId);
    to.incoming.add(fromId);
    this.edges.push(new GraphEdge(fromId, toId, condition));
  }

  getDependencies(nodeId) {
    const node = this.nodes.get(nodeId);
    return node ? Array.from(node.incoming) : [];
  }

  topologicalSort() {
    const inDegree = new Map();
    for (const [id] of this.nodes) inDegree.set(id, 0);
    for (const edge of this.edges) {
      inDegree.set(edge.to, (inDegree.get(edge.to) || 0) + 1);
    }

    const queue = [];
    for (const [id, deg] of inDegree) {
      if (deg === 0) queue.push(id);
    }

    const sorted = [];
    while (queue.length) {
      const id = queue.shift();
      sorted.push(id);
      const node = this.nodes.get(id);
      for (const nextId of node.outgoing) {
        const nextDeg = (inDegree.get(nextId) || 0) - 1;
        inDegree.set(nextId, nextDeg);
        if (nextDeg === 0) queue.push(nextId);
      }
    }
    return sorted;
  }
}

class SwarmRuntimeEngine {
  constructor() {
    this.actionQueue = [];
    this.skillCards = new Map();
    this.consensusDebates = [];
    this.graphs = new Map();

    this.registerDefaultSkills();
  }

  registerDefaultSkills() {
    this.addSkill(new SkillCard('skill-1', 'Neural Path Routing', 'Routes task across optimal subagents using Q-Learning & MCTS'));
    this.addSkill(new SkillCard('skill-2', 'Knowledge Extraction', 'Extracts ERD entities and builds relational graph'));
    this.addSkill(new SkillCard('skill-3', 'Code Refactoring', 'Performs multi-chunk file replacements and automated verification'));
    this.addSkill(new SkillCard('skill-4', 'Memory Consolidation', 'Distills ephemeral events into core wisdom nodes'));
  }

  addSkill(skillCard) {
    this.skillCards.set(skillCard.id, skillCard);
  }

  createGraph(name, nodes, edges) {
    const graph = new SwarmGraph(name);
    for (const node of nodes) {
      graph.addNode(node.id, node.action, node.loop || null);
    }
    for (const edge of edges) {
      graph.addEdge(edge.from, edge.to, edge.condition || null);
    }
    this.graphs.set(name, graph);
    return graph;
  }

  getGraph(name) {
    return this.graphs.get(name) || null;
  }

  async runWorkflow(name, context = {}) {
    const graph = this.graphs.get(name);
    if (!graph) return { status: 'failed', error: 'Workflow graph not found', name };

    const sorted = graph.topologicalSort();
    const results = new Map();
    const artifacts = [];
    const nodeExecutions = [];

    for (const nodeId of sorted) {
      const node = graph.nodes.get(nodeId);
      if (node.visited) continue;

      const depsReady = graph.getDependencies(nodeId).every(depId => {
        const dep = results.get(depId);
        return dep && dep.status === 'completed';
      });

      if (!depsReady) {
        return { status: 'failed', error: `Dependency not satisfied for node ${nodeId}`, name, results, nodeExecutions };
      }

      const startTs = Date.now();
      let nodeResult;
      if (node.loop) {
        const loopResult = await this.runLoop(
          node.loop,
          async (index, total) => {
            const loopOutput = {
              nodeId,
              loopIndex: index,
              loopTotal: total,
              actionType: node.action.type,
              target: node.action.target,
              parameters: { ...node.action.parameters, loopIndex: index, loopTotal: total },
              reasoning: `Loop iteration ${index + 1}/${total}`
            };
            artifacts.push(loopOutput);
            return loopOutput;
          },
          node.action.parameters?.step ? (i, total, ctx) => node.action.parameters.step(i, total, ctx) : null
        );
        node.visited = true;
        nodeResult = { status: 'completed', output: artifacts, loopResult };
      } else {
        const action = new SwarmAutonomyAction(
          `${nodeId}-${Date.now()}`,
          node.action.type,
          node.action.target,
          node.action.parameters,
          node.action.reasoning || `Workflow node ${nodeId} execution`
        );
        action.status = 'completed';
        node.output = action;
        node.visited = true;
        const artifact = { nodeId, actionId: action.id, status: action.status, output: action };
        artifacts.push(artifact);
        nodeResult = { status: 'completed', output: artifact };
      }

      results.set(nodeId, nodeResult);
      nodeExecutions.push({
        nodeId,
        status: nodeResult.status,
        latencyMs: Date.now() - startTs,
        reasoning: node.action.reasoning || null
      });

      if (typeof context.onNodeComplete === 'function') {
        context.onNodeComplete({
          nodeId,
          status: nodeResult.status,
          actionType: node.action.type,
          target: node.action.target,
          latencyMs: Date.now() - startTs,
          output: nodeResult.output
        });
      }
    }

    return { status: 'completed', results, artifacts, nodeExecutions, name };
  }

  enqueueAction(type, target, parameters = {}, reasoning = '') {
    const action = new SwarmAutonomyAction(null, type, target, parameters, reasoning);
    this.actionQueue.push(action);
    return action;
  }

  async runLoop(loopDef, iteratorFn, stepFn) {
    const { type, count, condition, maxIterations = 1000 } = loopDef || {};
    const safeCount = Math.min(Math.floor(count || 0), maxIterations);

    if (type === 'for' && safeCount > 0) {
      for (let i = 0; i < safeCount; i++) {
        if (stepFn) await stepFn(i, safeCount);
        if (iteratorFn) await iteratorFn(i, safeCount);
      }
      return { iterations: safeCount, status: 'completed' };
    }

    if (type === 'repeat' && safeCount > 0) {
      for (let i = 0; i < safeCount; i++) {
        if (stepFn) await stepFn(i, safeCount);
        if (iteratorFn) await iteratorFn(i, safeCount);
      }
      return { iterations: safeCount, status: 'completed' };
    }

    if (type === 'while') {
      let iterations = 0;
      while (condition && condition() && iterations < maxIterations) {
        if (stepFn) await stepFn(iterations, maxIterations);
        if (iteratorFn) await iteratorFn(iterations, maxIterations);
        iterations++;
      }
      return { iterations, status: iterations >= maxIterations ? 'maxed' : 'completed' };
    }

    return { iterations: 0, status: 'noop' };
  }

  async processGraph(name, context = {}) {
    const graph = this.graphs.get(name);
    if (!graph) return { status: 'failed', error: 'Graph not found' };

    const sorted = graph.topologicalSort();
    const results = new Map();

    for (const nodeId of sorted) {
      const node = graph.nodes.get(nodeId);
      if (node.visited) continue;

      const depsReady = graph.getDependencies(nodeId).every(depId => {
        const dep = results.get(depId);
        return dep && dep.status === 'completed';
      });

      if (!depsReady) {
        return { status: 'failed', error: `Dependency not satisfied for node ${nodeId}` };
      }

      if (node.loop) {
        const loopResult = await this.runLoop(
          node.loop,
          async (index, total) => {
            const loopAction = new SwarmAutonomyAction(
              `${nodeId}-loop-${index}`,
              node.action.type,
              node.action.target,
              { ...node.action.parameters, loopIndex: index, loopTotal: total },
              `Loop iteration ${index + 1}/${total} for node ${nodeId}`
            );
            loopAction.status = 'completed';
            node.output = { loopResult: loopResult, lastLoopAction: loopAction };
          },
          node.action.parameters?.step ? (i, total) => node.action.parameters.step(i, total, context) : null
        );
        node.visited = true;
        results.set(nodeId, { status: 'completed', output: node.output, loopResult: loopResult });
        continue;
      }

      const action = new SwarmAutonomyAction(
        `${nodeId}-${Date.now()}`,
        node.action.type,
        node.action.target,
        node.action.parameters,
        node.action.reasoning || `Graph node ${nodeId} execution`
      );
      action.status = 'completed';
      node.output = action;
      node.visited = true;
      results.set(nodeId, { status: 'completed', output: action });
    }

    return { status: 'completed', results, graph: name };
  }

  async processQueue(callback) {
    const processed = [];
    while (this.actionQueue.length > 0) {
      const action = this.actionQueue.shift();
      action.status = 'in_progress';
      
      // Simulate execution
      await new Promise(resolve => setTimeout(resolve, 300));

      action.status = 'completed';
      processed.push(action);

      if (typeof callback === 'function') {
        callback(action);
      }
    }
    return processed;
  }

  // Simulated Federated Multi-Agent Debate Consensus
  runDebateConsensus(topic, agentsList) {
    const debate = {
      id: `debate-${Date.now()}`,
      topic,
      timestamp: Date.now(),
      rounds: 3,
      consensusScore: 0.94,
      verdict: `Consensus reached by ${agentsList.length} agents: Proceed with parallel execution policy.`
    };
    this.consensusDebates.push(debate);
    return debate;
  }
}

if (typeof window !== 'undefined') {
  window.SwarmRuntime = {
    SkillCard,
    SwarmAutonomyAction,
    SwarmRuntimeEngine,
    SwarmGraph,
    GraphNode,
    GraphEdge
  };
}

if (typeof module !== 'undefined') {
  module.exports = {
    SkillCard,
    SwarmAutonomyAction,
    SwarmRuntimeEngine,
    SwarmGraph,
    GraphNode,
    GraphEdge
  };
}
