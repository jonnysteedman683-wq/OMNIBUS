/**
 * OMNIBUS Single Chat Interface & Backend Controller
 */
document.addEventListener('DOMContentLoaded', () => {
  const chatMessagesStream = document.getElementById('chatMessagesStream');
  const chatInputField = document.getElementById('chatInputField');
  const chatSendBtn = document.getElementById('chatSendBtn');
  const engineModeSelector = document.getElementById('engineModeSelector');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const orbStateLabel = document.getElementById('orbStateLabel');

  // Auto-resize textarea
  if (chatInputField) {
    chatInputField.addEventListener('input', () => {
      chatInputField.style.height = 'auto';
      chatInputField.style.height = Math.min(chatInputField.scrollHeight, 120) + 'px';
    });

    chatInputField.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
  }

  if (chatSendBtn) {
    chatSendBtn.addEventListener('click', handleSendMessage);
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', () => {
      if (chatMessagesStream) {
        chatMessagesStream.innerHTML = `
          <div class="message-card assistant">
            <div class="avatar assistant"><i class="fas fa-robot"></i></div>
            <div class="message-bubble">
              <div style="font-weight: 800; color: #00f0ff; margin-bottom: 4px;">OMNIBUS Core Assistant</div>
              Chat history cleared. System ready for new queries!
            </div>
          </div>
        `;
      }
    });
  }

  function setOrbState(state, labelText) {
    if (window.orbVisualizer) {
      window.orbVisualizer.setState(state);
    }
    if (orbStateLabel) {
      orbStateLabel.innerText = labelText || (state === 'thinking' ? 'Computing Real LLM Reasoning...' : 'OMNIBUS Neural Interface · Ready');
    }
  }

  function appendUserMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card user';
    card.innerHTML = `
      <div class="avatar user"><i class="fas fa-user"></i></div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    chatMessagesStream.appendChild(card);
    scrollToBottom();
  }

  function appendAssistantMessage(htmlContent, title = 'OMNIBUS Neural Core') {
    const card = document.createElement('div');
    card.className = 'message-card assistant';
    card.innerHTML = `
      <div class="avatar assistant"><i class="fas fa-microchip"></i></div>
      <div class="message-bubble">
        <div style="font-weight: 800; color: #00f0ff; margin-bottom: 6px;">${title}</div>
        <div>${htmlContent}</div>
      </div>
    `;
    chatMessagesStream.appendChild(card);
    scrollToBottom();
  }

  function scrollToBottom() {
    if (chatMessagesStream) {
      chatMessagesStream.scrollTop = chatMessagesStream.scrollHeight;
    }
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  async function handleSendMessage() {
    const promptText = chatInputField.value.trim();
    if (!promptText) return;

    chatInputField.value = '';
    chatInputField.style.height = 'auto';
    appendUserMessage(promptText);

    const mode = engineModeSelector ? engineModeSelector.value : 'v75_zenith';
    setOrbState('thinking', 'Executing v75.0 Zenith ML Core...');

    try {
      if (mode === 'v75_zenith') {
        const response = await fetch('/api/v75-frontier-zenith-synthesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: promptText })
        });
        const data = await response.json();
        setOrbState('responding', 'v75.0 Zenith Synthesis Complete');
        renderZenithV75Result(data.result);
      }
      else if (mode === 'v75_diffworld') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.DiffWorldLatentTrajectoryPlannerV75();
        const res = engine.sampleDenoisedTrajectory(null, [0.2, -0.5, 0.8]);
        setOrbState('responding', 'Diffusion World Model Rollout Complete');
        renderDiffWorldResult(res);
      }
      else if (mode === 'v75_selfevolve') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.SelfEvolvingRLVROptimizerV75();
        const res = engine.evaluateSelfEvolvingPass(promptText);
        setOrbState('responding', 'Self-Evolving DPO Alignment Complete');
        renderSelfEvolveResult(res);
      }
      else if (mode === 'v75_qtensor') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.QTensorNetMPSAttentionV75();
        const res = engine.factorizeAndCompressAttention();
        setOrbState('responding', 'Quantum MPS Tensor Compression Complete');
        renderQTensorResult(res);
      }
      else if (mode === 'v75_sparse_moe') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.SparseMoEGumbelRouterV75();
        const res = engine.routeTokens(promptText);
        setOrbState('responding', 'Sparse MoE Gumbel Routing Complete');
        renderSparseMoeResult(res);
      }
      else if (mode === 'v75_neuromorphic') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.NeuromorphicLiquidSNNEngineV75();
        const res = engine.stepSpikeDynamics();
        setOrbState('responding', 'Neuromorphic SNN Dynamics Complete');
        renderNeuromorphicResult(res);
      }
      else if (mode === 'v70_apex') {
        const response = await fetch('/api/v70-singularity-apex-synthesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: promptText })
        });
        const data = await response.json();
        setOrbState('responding', 'v70.0 Apex Synthesis Complete');
        renderApexV70Result(data.result);
      }
      else if (mode === 'v70_ttc') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.DynamicTestTimeComputeBudgetEngineV70();
        const res = engine.evaluateComplexityAndAllocateBudget(promptText);
        setOrbState('responding', 'Dynamic TTC Budget Allocated');
        renderTtcResult(res);
      }
      else if (mode === 'v70_rlvr') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.RLVRVerifiableRewardFeedbackEngineV70();
        const res = engine.evaluateGroupAdvantage(promptText);
        setOrbState('responding', 'RLVR Verification Complete');
        renderRlvrResult(res);
      }
      else if (mode === 'v70_swarm') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.MultiAgentSwarmDebateConsensusEngineV70(800);
        const res = engine.conductDebateRounds(promptText, 3);
        setOrbState('responding', 'Swarm Consensus Reached');
        renderSwarmDebateResult(res);
      }
      else if (mode === 'v70_speculative') {
        const expML = window.ExperimentalML || {};
        const engine = new expML.SelfSpeculativeDraftVerifierDecoderV70();
        const res = engine.runSpeculativeDecoding(promptText);
        setOrbState('responding', 'Speculative Verification Complete');
        renderSpeculativeResult(res);
      }
      else if (mode === 'prm_mcts') {
        const response = await fetch('/api/ml/real-prm-mcts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem: promptText, depth: 3 })
        });
        const data = await response.json();
        setOrbState('responding', 'Tree-Search Completed');
        renderPrmMctsResult(data.result);
      } 
      else if (mode === 'grpo_v3') {
        const response = await fetch('/api/ml/real-grpo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: promptText, candidateCount: 4 })
        });
        const data = await response.json();
        setOrbState('responding', 'GRPO Optimization Complete');
        renderGrpoResult(data.result);
      }
      else if (mode === 'titans') {
        if (promptText.toLowerCase().startsWith('store')) {
          const content = promptText.replace(/^store\s*/i, '');
          const response = await fetch('/api/titans/store', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key: "User Memory", content: content || promptText })
          });
          const data = await response.json();
          setOrbState('responding', 'Memory Stored');
          renderTitansStoreResult(data.result);
        } else {
          const response = await fetch('/api/titans/recall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: promptText, topK: 3 })
          });
          const data = await response.json();
          setOrbState('responding', 'Memory Recalled');
          renderTitansRecallResult(data.result);
        }
      }
      else if (mode === 'python_core') {
        let task = 'master';
        if (promptText.toLowerCase().includes('bitnet')) task = 'bitnet';
        else if (promptText.toLowerCase().includes('kan')) task = 'kan';
        else if (promptText.toLowerCase().includes('poincare')) task = 'poincare';
        else if (promptText.toLowerCase().includes('speculative')) task = 'speculative';
        else if (promptText.toLowerCase().includes('hopfield')) task = 'hopfield';
        else if (promptText.toLowerCase().includes('diffworld') || promptText.toLowerCase().includes('diffusion')) task = 'diffworld';
        else if (promptText.toLowerCase().includes('qtensor') || promptText.toLowerCase().includes('mps')) task = 'qtensornet';

        const response = await fetch('/api/python/ml-core', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task, params: { query: promptText } })
        });
        const data = await response.json();
        setOrbState('responding', 'Python Core Execution Complete');
        renderPythonResult(data.result);
      }

      setTimeout(() => {
        setOrbState('idle', 'OMNIBUS Neural Interface · v75.0 Zenith Ready');
      }, 1500);

    } catch (err) {
      setOrbState('idle', 'Error Encountered');
      appendAssistantMessage(`<span style="color: #ff0055;">Error: ${err.message}</span>`, 'System Alert');
    }
  }

  function renderPrmMctsResult(res) {
    if (!res) return;
    let pathHtml = res.chosenExecutionPath.map(step => `
      <div style="margin: 6px 0; padding: 8px 12px; background: rgba(0,240,255,0.06); border-left: 3px solid #00f0ff; border-radius: 4px;">
        <span style="color: #00f0ff; font-weight: bold;">[Step Depth ${step.depth}]</span> (Score: ${step.prmScore}): ${escapeHtml(step.stepContent)}
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Provider:</b> <span style="color: #00ff88; font-weight: bold;">${res.provider}</span></div>
        <div><b>Pass Rate:</b> <span style="color: #ff00ea; font-weight: bold;">${res.verifiedPassRate}</span></div>
        <div><b>Nodes Evaluated:</b> ${res.totalTreeNodesEvaluated} (Depth: ${res.treeDepth})</div>
        <div style="margin-top: 10px; font-weight: bold; color: #00f0ff;">Verified Optimal Path:</div>
        ${pathHtml}
      </div>
    `;
    appendAssistantMessage(html, '🌲 Real PRM-MCTS Tree Search Result');
  }

  function renderGrpoResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Provider:</b> <span style="color: #00ff88; font-weight: bold;">${res.provider}</span></div>
        <div><b>Group Size (K):</b> ${res.groupSize}</div>
        <div><b>Winning Advantage:</b> <span style="color: #ff00ea; font-weight: bold;">${res.winningCandidate?.advantagePercentage}</span></div>
        
        <div class="code-block-wrapper">
          <div class="code-block-header">Winning Reasoning Candidate (Temp = ${res.winningCandidate?.temperature})</div>
          <div class="code-block-content">${escapeHtml(res.winningCandidate?.solutionText || '')}</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🧮 Real GRPO-v3 Relative Advantage Result');
  }

  function renderTitansStoreResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Memory Status:</b> <span style="color: #00ff88; font-weight: bold;">STORED IN PERSISTENT VECTOR DB</span></div>
        <div><b>Surprise Gate Metric S:</b> ${res.memory?.surpriseMetric}</div>
        <div><b>Retention Score:</b> ${(res.memory?.retentionScore * 100).toFixed(2)}%</div>
        <div><b>Memory Key:</b> ${escapeHtml(res.memory?.key || '')}</div>
        <div><b>Content:</b> ${escapeHtml(res.memory?.content || '')}</div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Titans Surprise-Gated Memory Store');
  }

  function renderTitansRecallResult(res) {
    if (!res) return;
    let memoriesHtml = (res.recalledMemories || []).map(m => `
      <div style="margin: 6px 0; padding: 8px 12px; background: rgba(255,0,234,0.06); border-left: 3px solid #ff00ea; border-radius: 4px;">
        <div style="color: #ff00ea; font-weight: bold;">${escapeHtml(m.key)} (Similarity: ${(m.similarity * 100).toFixed(1)}%)</div>
        <div style="color: #e2e8f0; margin-top: 4px;">${escapeHtml(m.content)}</div>
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Query:</b> "${escapeHtml(res.query)}"</div>
        <div><b>Matches Found:</b> ${res.resultsCount}</div>
        <div style="margin-top: 10px;">${memoriesHtml || 'No matching vector memories found.'}</div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Titans Vector Memory Recall');
  }

  function renderPythonResult(res) {
    if (!res) return;
    let jsonFormatted = escapeHtml(JSON.stringify(res, null, 2));
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Python PyTorch/NumPy Tensor Core:</b></div>
        <div class="code-block-wrapper">
          <div class="code-block-header">Python Execution JSON Output</div>
          <div class="code-block-content">${jsonFormatted}</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🐍 Python PyTorch/NumPy Tensor Core Result');
  }

  function renderApexV70Result(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem; display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; gap: 10px; align-items: center;">
          <span style="background: rgba(0,240,255,0.15); border: 1px solid #00f0ff; color: #00f0ff; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.75rem;">${res.version}</span>
          <span style="color: #00ff88; font-weight: bold;">Status: ${res.status}</span>
        </div>
        <div><b>Active Swarm Agents:</b> ${res.activeSwarmAgents}</div>
        
        <div style="margin-top: 6px; padding: 8px 12px; background: rgba(0,240,255,0.05); border-left: 3px solid #00f0ff; border-radius: 4px;">
          <div style="color: #00f0ff; font-weight: bold;">🧠 Dynamic Test-Time Compute (TTC)</div>
          <div>Tier: ${res.testTimeCompute?.computeTier} | Token Budget: ${res.testTimeCompute?.thinkingBudgetTokens} tokens</div>
          <div>Depth: ${res.testTimeCompute?.allocatedDepth} | Breadth: ${res.testTimeCompute?.allocatedBreadth} | Efficiency: ${res.testTimeCompute?.prunedSearchEfficiency}</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(255,0,234,0.05); border-left: 3px solid #ff00ea; border-radius: 4px;">
          <div style="color: #ff00ea; font-weight: bold;">⚡ RLVR Verifiable Rewards</div>
          <div>Mean Reward: ${res.rlvrAdvantage?.meanGroupReward} | Top Candidate: ${res.rlvrAdvantage?.topCandidate?.verifiedStatus} (Advantage: +${res.rlvrAdvantage?.topCandidate?.grpoAdvantageScore})</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(0,255,136,0.05); border-left: 3px solid #00ff88; border-radius: 4px;">
          <div style="color: #00ff88; font-weight: bold;">🐝 Multi-Agent Swarm Debate (MAD-C)</div>
          <div>Consensus Confidence: ${res.multiAgentDebate?.finalConsensusConfidence} (${res.multiAgentDebate?.debateRoundsCompleted} rounds completed)</div>
          <div style="color: #e2e8f0; font-size: 0.82rem; margin-top: 2px;">${res.multiAgentDebate?.consensusDecision}</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(255,170,0,0.05); border-left: 3px solid #ffaa00; border-radius: 4px;">
          <div style="color: #ffaa00; font-weight: bold;">🔮 Self-Speculative Decoding & Hopfield Energy</div>
          <div>Draft Throughput Speedup: ${res.speculativeDecoding?.throughputSpeedup} | Hopfield Lyapunov Energy: ${res.hopfieldEnergyMemory?.lyapunovEnergyLevel}</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🚀 OMNIBUS v70.0 Singularity Apex Supreme Result');
  }

  function renderTtcResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Compute Tier:</b> <span style="color: #00f0ff; font-weight: bold;">${res.computeTier}</span></div>
        <div><b>Thinking Token Budget:</b> ${res.thinkingBudgetTokens} tokens</div>
        <div><b>Allocated Depth / Breadth:</b> Depth ${res.allocatedDepth}, Breadth ${res.allocatedBreadth}</div>
        <div><b>Verification Loops:</b> ${res.verificationLoops}</div>
        <div><b>Pruned Search Efficiency:</b> <span style="color: #00ff88; font-weight: bold;">${res.prunedSearchEfficiency}</span></div>
        <div><b>Tree PRM Confidence:</b> ${res.prmTreeConfidence}</div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Dynamic Test-Time Compute (TTC) Allocation');
  }

  function renderRlvrResult(res) {
    if (!res) return;
    let candidatesHtml = (res.allCandidates || []).map(c => `
      <div style="margin: 4px 0; padding: 6px 10px; background: rgba(0,255,136,0.04); border-left: 2px solid ${c.verifiedStatus.includes('PASSED') ? '#00ff88' : '#ff0055'}; border-radius: 4px;">
        <span style="font-weight: bold;">${c.id}</span> - Reward: ${c.rawReward} | Advantage: ${c.grpoAdvantageScore} | Status: ${c.verifiedStatus}
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Group Size (K):</b> ${res.groupSize}</div>
        <div><b>Mean Group Reward:</b> ${res.meanGroupReward}</div>
        <div><b>Reward Variance:</b> ${res.rewardVariance}</div>
        <div style="margin-top: 8px; font-weight: bold; color: #00ff88;">Evaluated Candidate Stream:</div>
        ${candidatesHtml}
      </div>
    `;
    appendAssistantMessage(html, '⚡ RLVR Verifiable Rewards Evaluation');
  }

  function renderSwarmDebateResult(res) {
    if (!res) return;
    let roundsHtml = (res.debateTrajectory || []).map(r => `
      <div style="margin: 4px 0; padding: 6px 10px; background: rgba(0,240,255,0.04); border-left: 2px solid #00f0ff; border-radius: 4px;">
        <b>Round ${r.round}</b> (${r.intermediateConsensus} consensus): ${escapeHtml(r.keyInsight)}
      </div>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Participating Swarm Agents:</b> ${res.totalSwarmAgents}</div>
        <div><b>Final Consensus Confidence:</b> <span style="color: #00ff88; font-weight: bold;">${res.finalConsensusConfidence}</span></div>
        <div><b>Decision:</b> ${escapeHtml(res.consensusDecision)}</div>
        <div style="margin-top: 8px; font-weight: bold; color: #00f0ff;">Debate Log:</div>
        ${roundsHtml}
      </div>
    `;
    appendAssistantMessage(html, '🐝 Multi-Agent Swarm Debate (MAD-C) Result');
  }

  function renderSpeculativeResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Draft K Lookahead:</b> ${res.draftKLookahead} tokens</div>
        <div><b>Accepted Tokens:</b> ${res.acceptedTokens}</div>
        <div><b>Acceptance Rate:</b> ${res.acceptanceRate}</div>
        <div><b>Throughput Acceleration:</b> <span style="color: #ffaa00; font-weight: bold;">${res.throughputSpeedup}</span></div>
        <div><b>Latency Reduction:</b> ${res.latencyReduction}</div>
      </div>
    `;
    appendAssistantMessage(html, '🔮 Self-Speculative Draft Decoding Result');
  }

  function renderZenithV75Result(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem; display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; gap: 10px; align-items: center;">
          <span style="background: rgba(0,240,255,0.15); border: 1px solid #00f0ff; color: #00f0ff; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.75rem;">${res.version}</span>
          <span style="color: #00ff88; font-weight: bold;">Status: ${res.status}</span>
        </div>
        <div><b>Active Frontier Engines / Swarm Agents:</b> ${res.activeFrontierMlEngines} / ${res.activeSwarmAgents}</div>
        
        <div style="padding: 8px 12px; background: rgba(0,240,255,0.05); border-left: 3px solid #00f0ff; border-radius: 4px;">
          <div style="color: #00f0ff; font-weight: bold;">🌌 Diffusion World Model Trajectory Rollout</div>
          <div>Steps Executed: ${res.diffWorldTrajectory?.diffusionStepsExecuted} | Final Latent Norm: ${res.diffWorldTrajectory?.finalLatentNorm} | Fidelity: ${res.diffWorldTrajectory?.trajectoryFidelity}</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(255,0,234,0.05); border-left: 3px solid #ff00ea; border-radius: 4px;">
          <div style="color: #ff00ea; font-weight: bold;">🧬 Self-Evolving DPO Alignment</div>
          <div>Winner Probe: ${res.selfEvolvingRLVR?.winnerProbeId} (Score: ${res.selfEvolvingRLVR?.winnerProbeScore}) | DPO Loss: ${res.selfEvolvingRLVR?.dpoLoss} | Pass Rate: ${res.selfEvolvingRLVR?.selfEvolvedPassRate}</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(0,255,136,0.05); border-left: 3px solid #00ff88; border-radius: 4px;">
          <div style="color: #00ff88; font-weight: bold;">⚛️ Quantum MPS Tensor Attention</div>
          <div>Compressed Memory: ${res.qTensorNetMPS?.mpsCompressedMemoryKb} KB vs ${res.qTensorNetMPS?.uncompressedMemoryKb} KB | Speedup: ${res.qTensorNetMPS?.compressionSpeedup} | Fidelity: ${res.qTensorNetMPS?.reconstructionFidelity}</div>
        </div>

        <div style="padding: 8px 12px; background: rgba(255,170,0,0.05); border-left: 3px solid #ffaa00; border-radius: 4px;">
          <div style="color: #ffaa00; font-weight: bold;">🔀 Sparse MoE Gumbel Router & Neuromorphic SNN</div>
          <div>Top-2 Experts Selected: ${res.sparseMoERouting?.selectedExperts?.map(e => `E${e.expertId}(${e.prob})`).join(', ')} | SNN Spikes Fired: ${res.neuromorphicLiquidSNN?.totalSpikesFired}/${res.neuromorphicLiquidSNN?.neuronCount} (${res.neuromorphicLiquidSNN?.energyEfficiencyJoulePerSpike})</div>
        </div>
      </div>
    `;
    appendAssistantMessage(html, '🌟 OMNIBUS v75.0 Singularity Zenith Supreme Result');
  }

  function renderDiffWorldResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Latent Dimension:</b> ${res.latentDimension}</div>
        <div><b>Diffusion Steps:</b> ${res.diffusionStepsExecuted}</div>
        <div><b>Final Latent Norm:</b> <span style="color: #00f0ff; font-weight: bold;">${res.finalLatentNorm}</span></div>
        <div><b>Denoising Confidence:</b> <span style="color: #00ff88; font-weight: bold;">${res.denoisingConfidence}</span></div>
        <div><b>Trajectory Fidelity:</b> ${res.trajectoryFidelity}</div>
      </div>
    `;
    appendAssistantMessage(html, '🌌 Diffusion World Trajectory Model Result');
  }

  function renderSelfEvolveResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Group Size:</b> ${res.groupSize} | <b>Beta DPO:</b> ${res.betaDPO}</div>
        <div><b>DPO Alignment Loss:</b> <span style="color: #ff00ea; font-weight: bold;">${res.dpoLoss}</span></div>
        <div><b>Winning Probe Candidate:</b> ${res.winnerProbeId} (Score: ${res.winnerProbeScore})</div>
        <div><b>Self-Evolved Pass Rate:</b> <span style="color: #00ff88; font-weight: bold;">${res.selfEvolvedPassRate}</span></div>
      </div>
    `;
    appendAssistantMessage(html, '🧬 Self-Evolving DPO Alignment Result');
  }

  function renderQTensorResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Sequence Length:</b> ${res.sequenceLength} | <b>Bond Dimension:</b> ${res.bondDimension}</div>
        <div><b>MPS Memory Reduction:</b> ${res.mpsCompressedMemoryKb} KB (from ${res.uncompressedMemoryKb} KB)</div>
        <div><b>Compression Speedup:</b> <span style="color: #00f0ff; font-weight: bold;">${res.compressionSpeedup}</span></div>
        <div><b>Reconstruction Fidelity:</b> <span style="color: #00ff88; font-weight: bold;">${res.reconstructionFidelity}</span></div>
      </div>
    `;
    appendAssistantMessage(html, '⚛️ Quantum MPS Tensor Attention Result');
  }

  function renderSparseMoeResult(res) {
    if (!res) return;
    let expertsHtml = (res.selectedExperts || []).map(e => `
      <span style="display: inline-block; margin: 2px 4px; padding: 2px 8px; background: rgba(255,170,0,0.15); border: 1px solid #ffaa00; border-radius: 12px; color: #ffaa00; font-size: 0.8rem; font-weight: bold;">
        Expert #${e.expertId} (p=${e.prob})
      </span>
    `).join('');

    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Total Experts:</b> ${res.totalExperts} | <b>Top-K Selected:</b> ${res.topKSelected}</div>
        <div><b>Selected Gumbel Experts:</b> ${expertsHtml}</div>
        <div><b>Auxiliary Sinkhorn Load Loss:</b> ${res.auxiliaryLoadBalanceLoss}</div>
        <div><b>Routing Efficiency:</b> <span style="color: #00ff88; font-weight: bold;">${res.routingEfficiency}</span></div>
      </div>
    `;
    appendAssistantMessage(html, '🔀 Sparse MoE Gumbel Router Result');
  }

  function renderNeuromorphicResult(res) {
    if (!res) return;
    let html = `
      <div style="font-size: 0.9rem;">
        <div><b>Engine:</b> ${res.engine}</div>
        <div><b>Neurons / Membrane Tau:</b> ${res.neuronCount} neurons / ${res.tauMembraneMs} ms</div>
        <div><b>Total Spikes Fired:</b> ${res.totalSpikesFired} (${res.meanFiringRate} firing rate)</div>
        <div><b>STDP Weight Delta:</b> ${res.stdpSynapticWeightDelta}</div>
        <div><b>Energy Consumption:</b> <span style="color: #00ff88; font-weight: bold;">${res.energyEfficiencyJoulePerSpike}</span></div>
      </div>
    `;
    appendAssistantMessage(html, '🧠 Neuromorphic Liquid SNN Result');
  }
});
