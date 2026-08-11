/**
 * OMNIBUS Single Chat Interface & Backend Controller
 */
// Hive Swarm Mind — global API config
window.apiConfig = window.apiConfig || { provider: 'hermes', model: 'hermes3' };

function hashIntentPhase(intent, source = 'hermes') {
  const str = `${source}:${intent}`;
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h * 31 + str.charCodeAt(i)) | 0;
  }
  return ((h % 8) + 8) % 8;
}

// Initialize Neurocore connection on load
async function initHiveSwarmMind() {
  try {
    const res = await fetch('/api/neurocore/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        baseUrl: 'http://localhost:8080/v1',
        enableHermes: true
      })
    });
    const data = await res.json();
    if (data.success) {
      window.omnibusSwarmConnected = true;
      if (window.orbVisualizer) {
        window.orbVisualizer.setState('idle');
      }
    }
    
    // Start status polling
    updateHiveSwarmStatus();
    setInterval(updateHiveSwarmStatus, 5000);
    updateHiveMemoryLearningWidget();
    setInterval(updateHiveMemoryLearningWidget, 7000);
  } catch (err) {
    console.warn('[app.js] Hive Swarm Mind connection failed:', err.message);
    window.omnibusSwarmConnected = false;
    updateHiveSwarmStatus();
  }
}

async function updateHiveSwarmStatus() {
  const statusEl = document.getElementById('hiveSwarmStatus');
  const textEl = document.getElementById('hiveSwarmStatusText');
  if (!statusEl || !textEl) return;

  try {
    const res = await fetch('/api/neurocore/status');
    const data = await res.json();
    if (data.success && data.connected) {
      statusEl.style.display = 'inline-flex';
      textEl.textContent = `Hive Swarm Mind: ${data.lastProvider || 'ready'} · ${data.peers?.length || 0} peers`;
      statusEl.style.background = 'rgba(0, 255, 136, 0.12)';
      statusEl.style.borderColor = 'rgba(0, 255, 136, 0.3)';
      statusEl.style.color = '#00ff88';
    } else if (data.neurocoreAvailable) {
      statusEl.style.display = 'inline-flex';
      textEl.textContent = 'Hive Swarm Mind: available but disconnected';
      statusEl.style.background = 'rgba(255, 187, 0, 0.12)';
      statusEl.style.borderColor = 'rgba(255, 187, 0, 0.3)';
      statusEl.style.color = '#ffbb00';
    } else {
      statusEl.style.display = 'inline-flex';
      textEl.textContent = 'Hive Swarm Mind: unavailable';
      statusEl.style.background = 'rgba(255, 60, 60, 0.12)';
      statusEl.style.borderColor = 'rgba(255, 60, 60, 0.3)';
      statusEl.style.color = '#ff3c3c';
    }

    // Update queue widget
    const queueEl = document.getElementById('hiveQueueWidget');
    const queueTextEl = document.getElementById('hiveQueueText');
    if (queueEl && queueTextEl) {
      const queueSize = typeof data.queueSize === 'number' ? data.queueSize : null;
      if (queueSize !== null && data.success && data.connected) {
        queueEl.style.display = 'inline-flex';
        queueTextEl.textContent = `Queue: ${queueSize}`;
      } else {
        queueEl.style.display = 'none';
      }
    }
  } catch (err) {
    statusEl.style.display = 'inline-flex';
    textEl.textContent = 'Hive Swarm Mind: error';
    statusEl.style.background = 'rgba(255, 60, 60, 0.12)';
    statusEl.style.borderColor = 'rgba(255, 60, 60, 0.3)';
    statusEl.style.color = '#ff3c3c';
  }
}

async function updateHiveMemoryLearningWidget() {
  const memoryEl = document.getElementById('hiveMemoryStatus');
  const learningEl = document.getElementById('hiveLearningStatus');
  if (!memoryEl && !learningEl) return;

  try {
    const [memoryRes, learningRes] = await Promise.all([
      fetch('/api/neurocore/memory'),
      fetch('/api/neurocore/learning')
    ]);
    const memoryData = await memoryRes.json();
    const learningData = await learningRes.json();

    if (memoryEl) {
      const recentCount = Array.isArray(memoryData.recent) ? memoryData.recent.length : 0;
      const failedCount = typeof memoryData.failedCount === 'number' ? memoryData.failedCount : 0;
      memoryEl.textContent = `Memory: ${recentCount} recent · ${failedCount} failed`;
      memoryEl.style.display = 'inline-flex';
      memoryEl.style.background = memoryData.connected ? 'rgba(0, 255, 136, 0.12)' : 'rgba(255, 187, 0, 0.12)';
      memoryEl.style.borderColor = memoryData.connected ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 187, 0, 0.3)';
      memoryEl.style.color = memoryData.connected ? '#00ff88' : '#ffbb00';
    }

    if (learningEl) {
      const stats = Array.isArray(learningData.providerStats) ? learningData.providerStats : [];
      const top = stats.slice(0, 2).map(s => `${s.provider}:${(s.successRate * 100).toFixed(0)}%`).join(' · ') || 'no data';
      learningEl.textContent = `Learning: ${top}`;
      learningEl.style.display = 'inline-flex';
      learningEl.style.background = learningData.connected ? 'rgba(0, 240, 255, 0.12)' : 'rgba(255, 187, 0, 0.12)';
      learningEl.style.borderColor = learningData.connected ? 'rgba(0, 240, 255, 0.3)' : 'rgba(255, 187, 0, 0.3)';
      learningEl.style.color = learningData.connected ? '#00f0ff' : '#ffbb00';
    }
  } catch (err) {
    if (memoryEl) {
      memoryEl.textContent = 'Memory: error';
      memoryEl.style.display = 'inline-flex';
      memoryEl.style.background = 'rgba(255, 60, 60, 0.12)';
      memoryEl.style.borderColor = 'rgba(255, 60, 60, 0.3)';
      memoryEl.style.color = '#ff3c3c';
    }
    if (learningEl) {
      learningEl.textContent = 'Learning: error';
      learningEl.style.display = 'inline-flex';
      learningEl.style.background = 'rgba(255, 60, 60, 0.12)';
      learningEl.style.borderColor = 'rgba(255, 60, 60, 0.3)';
      learningEl.style.color = '#ff3c3c';
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const chatMessagesStream = document.getElementById('chatMessagesStream');
  const chatInputField = document.getElementById('chatInputField');
  const chatSendBtn = document.getElementById('chatSendBtn');
  const engineModeSelector = document.getElementById('engineModeSelector');
  const providerSelector = document.getElementById('providerSelector');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const orbStateLabel = document.getElementById('orbStateLabel');

  // Auto-init Hive Swarm Mind
  initHiveSwarmMind();


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

    const mode = engineModeSelector ? engineModeSelector.value : 'v200_omnipresent';
    try {
      if (mode === 'v200_omnipresent') {
        setOrbState('thinking', 'Executing v200.0 Singularity Omnipresent Apex Master ML Core...');
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: promptText })
        });
        const data = await response.json();
        setOrbState('responding', 'v200.0 Omnipresent Apex Synthesis Complete');
        appendAssistantMessage(escapeHtml(typeof data.reply === 'string' ? data.reply : JSON.stringify(data)), 'OMNIBUS Neural Core');
      }
      else if (mode === 'python_core') {
        if (window.omnibusSwarmConnected) {
          setOrbState('thinking', '🐝 Hive Swarm Mind: SafetyGate → Debate → Execute...');
          const agentSystem = window.OMNIBUS ? new window.OMNIBUS() : null;
          if (agentSystem && agentSystem.dispatch) {
            const result = await agentSystem.dispatch({
              description: promptText,
              confidence: 0.5,
              features: { mode, prompt: promptText },
              requiresConfirmation: false
            });
            if (result) {
              appendAssistantMessage(escapeHtml(result.response), '🐝 Hive Swarm Mind');
              setTimeout(() => {
                setOrbState('idle', 'OMNIBUS Neural Interface · v85.0 Singularity Nexus Ready');
              }, 1500);
              return;
            }
          }
        }
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
      else {
        setOrbState('thinking', 'Routing swarm intent through unified dispatcher...');
        const res = await fetch('/api/neurocore/intent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ intent: promptText, source: 'ui', confidence: 0.9, features: {}, requiresConfirmation: false })
        });
        const data = await res.json();
        setOrbState('responding', 'Swarm Dispatch Complete');
        appendAssistantMessage(`<div class="mono">Provider: ${data?.triage?.routedTo || 'hermes'}</div><div class="mono">Confidence: ${data?.triage?.confidence ?? '—'}</div><div>Status: ${data?.status || 'completed'}</div><div class="mono">actionId: ${data?.actionId || '—'}</div>`, 'Swarm Dispatch');
      }

      setTimeout(() => {
        setOrbState('idle', 'OMNIBUS Neural Interface · v85.0 Singularity Nexus Ready');
      }, 1500);
    } catch (err) {
      setOrbState('idle', 'Error Encountered');
      appendAssistantMessage(`<span style="color: #ff0055;">Error: ${err.message}</span>`, 'System Alert');
    }
  }











































































































  // ─── v95.0 Singularity Omniverse Result Renderers ────────────────────















  // ─── v150.0 Render Functions ─────────────────────────────────────────────








  // ─── ML Frontier & Architecture Innovation Lab Modal & Topology Visualizer ───
  const openMlBrainstormModalBtn = document.getElementById('openMlBrainstormModalBtn');
  const closeMlBrainstormModalBtn = document.getElementById('closeMlBrainstormModalBtn');
  const mlBrainstormModal = document.getElementById('mlBrainstormModal');
  const runBrainstormSynthesisBtn = document.getElementById('runBrainstormSynthesisBtn');
  const injectBrainstormToChatBtn = document.getElementById('injectBrainstormToChatBtn');
  const mlBrainstormResults = document.getElementById('mlBrainstormResults');
  const mlTopologyCanvas = document.getElementById('mlTopologyCanvas');

  let animFrameId = null;

  if (openMlBrainstormModalBtn && mlBrainstormModal) {
    openMlBrainstormModalBtn.addEventListener('click', () => {
      mlBrainstormModal.classList.add('active');
      startTopologyVisualization();
    });
  }

  if (closeMlBrainstormModalBtn && mlBrainstormModal) {
    closeMlBrainstormModalBtn.addEventListener('click', () => {
      mlBrainstormModal.classList.remove('active');
      if (animFrameId) cancelAnimationFrame(animFrameId);
    });
  }

  if (runBrainstormSynthesisBtn) {
    runBrainstormSynthesisBtn.addEventListener('click', async () => {
      if (mlBrainstormResults) {
        mlBrainstormResults.style.display = 'block';
        mlBrainstormResults.innerText = '⚡ Synthesizing & Benchmarking Brainstormed v100000.0 Architecture across 32,768-D Manifold...';
      }
      try {
        const response = await fetch('/api/v100000-singularity-transcendent-hypermind-synthesis', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: "Custom Brainstormed ML Architecture Benchmark Suite" })
        });
        const data = await response.json();
        if (data.success && mlBrainstormResults) {
          const r = data.result;
          mlBrainstormResults.innerText = JSON.stringify({
            architectureVersion: r.version,
            status: r.status,
            confidenceScore: r.transcendentHyperMindConfidenceScore,
            performance: r.performanceMetrics,
            selectedModules: {
              s12SymplecticSSM: document.getElementById('modS12')?.checked,
              titans1000TTTMind: document.getElementById('modTitans1000')?.checked,
              subBit000000000001bMoDMoE: document.getElementById('modSubBit')?.checked,
              cfmContinuousDiffMCTS: document.getElementById('modCFM')?.checked,
              swarmRLVRTheoremProver: document.getElementById('modSwarmRLVR')?.checked,
              vietorisRipsTDAGuard: document.getElementById('modTDABetti')?.checked,
              quantumPhaseVSA: document.getElementById('modQuantumVSA')?.checked,
              waveletKANMLA: document.getElementById('modWaveletMLA')?.checked
            }
          }, null, 2);
        }
      } catch (err) {
        if (mlBrainstormResults) {
          mlBrainstormResults.innerText = "Error: " + err.message;
        }
      }
    });
  }

  if (injectBrainstormToChatBtn && mlBrainstormModal) {
    injectBrainstormToChatBtn.addEventListener('click', () => {
      const chatInput = document.getElementById('chatInputField');
      if (chatInput) {
        chatInput.value = "Execute synthesis of brainstormed v100000.0 Transcendent Hyper-Mind architecture with S12 Symplectic SSM, Titans-v1000 TTT, 0.000000000001-Bit Entropic Router & Swarm RLVR Prover.";
        mlBrainstormModal.classList.remove('active');
        if (animFrameId) cancelAnimationFrame(animFrameId);
        document.getElementById('chatSendBtn')?.click();
      }
    });
  }

  function startTopologyVisualization() {
    if (!mlTopologyCanvas) return;
    const ctx = mlTopologyCanvas.getContext('2d');
    let t = 0;

    const nodes = [
      { x: 70, y: 110, label: "S13 Kahler SSM", color: "#00f0ff" },
      { x: 190, y: 60, label: "Titans v10000 TTT", color: "#ff00ea" },
      { x: 320, y: 160, label: "Sub-Bit 0.0000000000001b MoE", color: "#00ff88" },
      { x: 460, y: 70, label: "CFM Kinetic Diff-MCTS", color: "#ffd700" },
      { x: 600, y: 150, label: "Swarm RLVR GRPO", color: "#00f0ff" },
      { x: 730, y: 60, label: "TDA Betti-Guard", color: "#ff00ea" },
      { x: 860, y: 150, label: "Quantum Phase VSA", color: "#00ff88" },
      { x: 950, y: 90, label: "Wavelet KAN MLA", color: "#ffd700" }
    ];

    function draw() {
      t += 0.03;
      ctx.clearRect(0, 0, mlTopologyCanvas.width, mlTopologyCanvas.height);

      // Draw connecting energy pulses
      for (let i = 0; i < nodes.length - 1; i++) {
        const n1 = nodes[i];
        const n2 = nodes[i + 1];

        ctx.beginPath();
        ctx.moveTo(n1.x, n1.y);
        ctx.lineTo(n2.x, n2.y);
        ctx.strokeStyle = "rgba(0, 240, 255, 0.25)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Pulsing energy particle
        const progress = (t + i * 0.4) % 1;
        const px = n1.x + (n2.x - n1.x) * progress;
        const py = n1.y + (n2.y - n1.y) * progress;

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = n1.color;
        ctx.shadowColor = n1.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Draw nodes
      nodes.forEach((n, idx) => {
        const pulse = Math.sin(t * 2 + idx) * 3;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 14 + pulse, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(10, 14, 28, 0.9)";
        ctx.strokeStyle = n.color;
        ctx.lineWidth = 3;
        ctx.shadowColor = n.color;
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;

        ctx.fillStyle = "#ffffff";
        ctx.font = "600 11px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(n.label, n.x, n.y + 32);
      });

      animFrameId = requestAnimationFrame(draw);
    }

    draw();
  }
});
